import torch

from .utils import smooth

NUM_NODE = 9
NODE_SHAPE = (3, 3)
ZERO_INDEX = (1, 1)
SAMPLE_PER_TRAJECTORY = 10

def initialize(
    num_node=NUM_NODE,
    node_shape=NODE_SHAPE,
    zero_index=ZERO_INDEX,
    off_std=0.2,
    connection_limit=1.3, # sqrt(2) + eps
    sample_per_trajectory=SAMPLE_PER_TRAJECTORY
):
    node_indices = torch.arange(num_node).view(*node_shape)

    nodes = initialize_nodes(node_shape, zero_index, off_std)
    edges = initialize_edges(nodes, connection_limit)
    edge_trajs = initialize_edge_trajectories(edges, nodes, sample_per_trajectory)
    breakpoint()



def initialize_nodes(node_shape: "tuple[int, int]", zero_index: "tuple[int, int]", off_std: float) -> torch.Tensor:
    nodes = []
    for i in range(node_shape[0]):
        for j in range(node_shape[1]):
            node_ij = torch.tensor([i-zero_index[0], j-zero_index[1]], dtype=torch.float)
            node_ij += torch.zeros_like(node_ij).normal_(0, off_std)
            nodes.append(node_ij)
    nodes = torch.stack(nodes)

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
    edge_trajectory = torch.zeros([*edges.shape, sample_per_trajectory])
    for i in range(edges.shape[0]):
        for j in range(i+1, edges.shape[0]):
            if edges[i, j]:
                traj = smooth(nodes[i][0], nodes[j][0], sample_per_trajectory + 2)[1:-1]
                # + 2 and trimming both edge because smooth result is closed range
                edge_trajectory[i, j] = traj
                edge_trajectory[j, i] = torch.flip(traj, (0,))
    return edge_trajectory



if __name__ == "__main__":
    initialize()
