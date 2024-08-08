import torch

from .utils import smooth

NODE_SHAPE = (3, 3)
ZERO_INDEX = (1, 1)
SAMPLE_PER_TRAJECTORY = 10
NUM_LANDMARK = 3

def get_preset(
    node_shape=NODE_SHAPE,
    zero_index=ZERO_INDEX,
    off_std=0.2,
    connection_limit=1.5, # sqrt(2) + eps
    sample_per_trajectory=SAMPLE_PER_TRAJECTORY,
    speaking=False
):
    nodes = initialize_nodes(node_shape, zero_index, speaking, off_std)
    edges = initialize_edges(nodes, connection_limit)
    edge_trajs = initialize_edge_trajectories(edges, nodes, sample_per_trajectory)

    idx_nodes, idx_edges, motion_landmarks = get_motion_indices(nodes, edges, edge_trajs)

    return nodes, edges, idx_nodes, idx_edges, motion_landmarks


def initialize_nodes(
    node_shape: "tuple[int, int]",
    zero_index: "tuple[int, int]",
    speaking: bool,
    off_std: float) -> torch.Tensor:
    nodes = []
    for i in range(node_shape[0]):
        for j in range(node_shape[1]):
            node_ij = torch.tensor([i-zero_index[0], j-zero_index[1]], dtype=torch.float)
            node_ij += torch.zeros_like(node_ij).normal_(0, off_std)
            nodes.append(node_ij)
    nodes = torch.stack(nodes)

    mean_mouth = 0.27 if speaking else 0.05
    mouth = torch.zeros([len(nodes), 1]).normal_(mean_mouth, 0.04).clamp(min=0.01)
    nodes = torch.cat((nodes, mouth), 1)

    return nodes

def initialize_edges(nodes, connection_limit, num_trial=100):
    i = 0
    while invalid_edges(edges:=get_edges(nodes, connection_limit)):
        i+=1
        if i >= num_trial:
            raise ValueError("there are separate subgraphs or no-edge nodes. Larger connection_limit needed")
    return edges

def invalid_edges(edges):
    condition1 = edges.sum(0).all().eq(False).item() # there is at least one no-edge-node
    condition2 = has_unreachable(edges) # floyd_warshall

    return condition1 or condition2

def has_unreachable(edges):
    n = len(edges)
    reach = [[edges[i][j] != 0 for j in range(n)] for i in range(n)]

    # Set the diagonal to True
    for i in range(n):
        reach[i][i] = True

    # Floyd-Warshall algorithm to update reachability
    for k in range(n):
        for i in range(n):
            for j in range(n):
                reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])

    # Check if all pairs are reachable
    for i in range(n):
        for j in range(n):
            if not reach[i][j]:
                return True
    return False

def get_edges(nodes, connection_limit):
    distances = torch.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            dist_ij = torch.linalg.norm(nodes[i]-nodes[j], 2)
            distances[i, j] = dist_ij
            distances[j, i] = dist_ij
    edges = distances.lt(connection_limit)
    for i in range(len(nodes)):
        edges[i, i] = False
    return edges

def initialize_edge_trajectories(edges, nodes, sample_per_trajectory):
    edge_trajectory = torch.zeros([*edges.shape, sample_per_trajectory, NUM_LANDMARK])
    for i in range(edges.shape[0]):
        for j in range(i+1, edges.shape[0]):
            if edges[i, j]:
                traj = torch.cat([smooth(nodes[i][k], nodes[j][k], sample_per_trajectory + 2)[1:-1].unsqueeze(-1) for k in range(NUM_LANDMARK)], -1)
                # + 2 and trimming both edge because smooth result is closed range
                edge_trajectory[i, j] = traj
                edge_trajectory[j, i] = torch.flip(traj, (0,))
    return edge_trajectory

def get_motion_indices(nodes, edges, edge_trajectories):
    motion_landmarks = [nodes[i] for i in range(len(nodes))]

    # fetch motion landmarks
    idx = len(nodes)
    for i in range(len(edges)):
        for j in range(i+1, len(edges)):
            if edges[i, j]:
                motion_landmarks.extend(list(edge_trajectories[i, j]))
                idx += len(edge_trajectories)
    motion_landmarks = torch.stack(motion_landmarks)

    # match landmark indices to nodes, edges
    lm_indices_nodes = torch.arange(len(nodes))
    lm_indices_edges = torch.zeros(edge_trajectories.shape[:-1], dtype=torch.int) - 1
    idx = len(nodes)
    for i in range(len(edges)):
        for j in range(i+1, len(edges)):
            if edges[i, j]:
                len_edge_trajectories = edge_trajectories.shape[-2]
                upper = torch.arange(len_edge_trajectories)+idx
                lower = torch.flip(upper, (0,))
                lm_indices_edges[i, j] = upper
                lm_indices_edges[j, i] = lower
                idx += len_edge_trajectories

    # algorithm sanity check
    for idx in lm_indices_nodes:
        assert motion_landmarks[idx].eq(nodes[idx]).all()
    for i in range(len(edges)):
        for j in range(len(edges)):
            if lm_indices_edges[i, j, 0] == -1:
                continue
            for k in range(len(lm_indices_edges[i, j])):
                assert motion_landmarks[lm_indices_edges[i, j, k]].eq(edge_trajectories[i, j, k]).all()

    return lm_indices_nodes, lm_indices_edges, motion_landmarks




if __name__ == "__main__":
    # import numpy as np
    # time = np.arange(5*25) / 20
    # mouth_seq = (0.1*np.sin(time*11) + 0.031*np.sin(time*15)+0.01*np.sin(time*30)+0.171).astype(float)
    # breakpoint()
    preset_non_speaking = get_preset()
    preset_speaking = get_preset(speaking=True)

    is_speaking = torch.where(torch.linspace(0, 5, 100).sin().abs() < 0.5, 1, 0)

    breakpoint()
