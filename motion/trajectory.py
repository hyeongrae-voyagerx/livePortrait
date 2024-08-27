import torch
import torchaudio
from .preset import get_preset, initialize_edge_trajectories, get_motion_indices, SAMPLE_PER_TRAJECTORY

from random import sample

SPEAKING_CHANGE_CHECK = SAMPLE_PER_TRAJECTORY // 2
NUM_TRANSFER_SAMPLE = 10

class Trajectory:
    def __init__(self, start_idx, init_eye=0.35, speaking_change_check=SPEAKING_CHANGE_CHECK, fps=25):
        self.start_idx = start_idx
        self.speaking_change_check = speaking_change_check
        self.fps = fps

        self.speaking_graph = get_preset(speaking=True)
        self.non_speaking_graph = get_preset(speaking=False)

        self.merge_graphs()

    def merge_graphs(self):
        self.graph = {}
        nodes = torch.cat([self.non_speaking_graph["nodes"], self.speaking_graph["nodes"]], 0)
        edges = self.parallel_merge_edges(self.non_speaking_graph["edges"], self.speaking_graph["edges"])

        edge_trajectories = initialize_edge_trajectories(edges, nodes, SAMPLE_PER_TRAJECTORY)
        node_idx, edge_idx, landmarks = get_motion_indices(nodes, edges, edge_trajectories)

        self.graph = {
            "nodes": nodes,
            "edges": edges,
            "node_idx": node_idx,
            "edge_idx": edge_idx,
            "landmarks": landmarks
        }
        self.non_speaking_mask = self.generate_mask(edges, remain="UL")
        self.speaking_mask = self.generate_mask(edges, remain="DR")
        self.transfer_ns2s_mask = self.generate_mask(edges, remain="UR")
        self.transfer_s2ns_mask = self.generate_mask(edges, remain="DL")

        self.num_ns_nodes = len(nodes) // 2

    def parallel_merge_edges(self, edges1, edges2):
        edges1 = torch.cat([edges1, torch.eye(len(edges1))], 1)
        edges2 = torch.cat([torch.eye(len(edges2)), edges2], 1)

        return torch.cat([edges1, edges2], 0)

    def generate_mask(self, edges, remain="UL"): # U: Upper, D: Down, L: Left, R: Right
        half = edges.shape[0] // 2

        if remain == "UL":
            upper = torch.cat((torch.ones([half, half], dtype=torch.bool), torch.zeros([half, half], dtype=torch.bool)), 1)
            lower = torch.cat((torch.zeros([half, half], dtype=torch.bool), torch.zeros([half, half], dtype=torch.bool)), 1)
        elif remain == "UR":
            upper = torch.cat((torch.zeros([half, half], dtype=torch.bool), torch.ones([half, half], dtype=torch.bool)), 1)
            lower = torch.cat((torch.zeros([half, half], dtype=torch.bool), torch.zeros([half, half], dtype=torch.bool)), 1)
        elif remain == "DL":
            upper = torch.cat((torch.zeros([half, half], dtype=torch.bool), torch.zeros([half, half], dtype=torch.bool)), 1)
            lower = torch.cat((torch.ones([half, half], dtype=torch.bool), torch.zeros([half, half], dtype=torch.bool)), 1)
        elif remain == "DR":
            upper = torch.cat((torch.zeros([half, half], dtype=torch.bool), torch.zeros([half, half], dtype=torch.bool)), 1)
            lower = torch.cat((torch.zeros([half, half], dtype=torch.bool), torch.ones([half, half], dtype=torch.bool)), 1)

        mask = torch.cat((upper, lower), 0)

        return mask

    def __call__(self, speaking):
        return self.generate_trajectory(speaking)

    def generate_trajectory(self, speaking):
        seek = 0
        now = self.start_idx + self.num_ns_nodes
        trajectory = [now]
        while seek < len(speaking):
            is_changed = seek and speaking_now != speaking[seek]
            speaking_now = speaking[seek]
            if not is_changed or self.keeping(speaking[seek:], self.is_speaking_node(now)):
                next_node = self.sample_next_node(now, transfer=False)
                edge_trajectory = self.get_edge_trajectory(now, next_node)
                trajectory.extend(edge_trajectory.tolist())
                trajectory.append(next_node)
            else:
                next_node = self.sample_next_node(now, transfer=True)
                edge_trajectory = self.get_edge_trajectory(now, next_node)
                trajectory.extend(edge_trajectory.tolist())
                trajectory.append(next_node)

            seek = len(trajectory)-1
            now = next_node
        trajectory = trajectory[:len(speaking)]
        return trajectory

    def sample_next_node(self, now, transfer):
        if transfer:
            mask = self.transfer_s2ns_mask if self.is_speaking_node(now) else self.transfer_ns2s_mask
        else:
            mask = self.speaking_mask if self.is_speaking_node(now) else self.non_speaking_mask
        edges = mask * self.graph["edges"]
        candidates = [i for i in range(len(edges)) if edges[now, i]]
        next_node = sample(candidates, 1)[0]
        return next_node

    def get_edge_trajectory(self, now, next_node):
        return self.graph["edge_idx"][now, next_node]


    def process_wav(self, path, verbose=False):
        wav, sr = self.get_wav(path)
        seconds = wav.shape[1] / sr
        num_frames = int(seconds * self.fps)
        if verbose:
            print(f"| Duration {seconds:.3f} seconds wav file")
            print(f"| The number of frames: {num_frames}")

        return wav, sr, num_frames, seconds

    def is_speaking_node(self, node_idx):
        return self.num_ns_nodes <= node_idx < 2*self.num_ns_nodes

    @staticmethod
    def get_wav(path):
        wav, sr = torchaudio.load(path)
        return wav, sr

    @staticmethod
    def keeping(seq:torch.Tensor, now:int):
        to_see = seq[:SPEAKING_CHANGE_CHECK]
        return all(item==now for item in to_see.tolist())



if __name__ == "__main__":
    is_speaking = torch.where(torch.linspace(0, 5.3, 125).sin().abs() < 0.5, 1, 0)
    from .audio import AudioProcessor

    a = AudioProcessor()
    is_speaking = a("en_example.wav").to(torch.int)


    t = Trajectory(0)
    trajectory = t(is_speaking)
    breakpoint()
