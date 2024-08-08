import torch
import torchaudio
from .preset import get_preset, SAMPLE_PER_TRAJECTORY
from .utils import smooth

from random import sample

SPEAKING_CHANGE_CHECK = SAMPLE_PER_TRAJECTORY // 2
NUM_TRANSFER_SAMPLE = 10

class Trajectory:
    def __init__(self, start_idx, speaking_change_check=SPEAKING_CHANGE_CHECK, fps=25):
        self.start_idx = start_idx
        self.speaking_change_check = speaking_change_check
        self.fps = fps
        self.preset = get_preset()
        self.preset_speaking = list(get_preset(speaking=True))
        num_last = self.preset[3].max() + 1
        self.preset_speaking[2] += num_last
        self.preset_speaking[3] = torch.where(self.preset_speaking[3].eq(-1), -1, self.preset_speaking[3]+num_last)
        self.preset_speaking[4] = torch.cat((torch.zeros([num_last, self.preset_speaking[4].shape[-1]]), self.preset_speaking[4]), 0)
        self.preset_speaking[4][:num_last] = self.preset[4]

        transfer_traj = []
        for i in range(len(self.preset[0])):
            ns_node_i = self.preset[0][i]
            s_node_i  = self.preset_speaking[0][i]
            traj = torch.cat([smooth(ns_node_i[k], s_node_i[k], NUM_TRANSFER_SAMPLE + 2)[1:-1].unsqueeze(-1) for k in range(self.preset_speaking[4].shape[-1])], -1)
            transfer_traj.append(traj)
        self.transfer_trajectory = torch.stack(transfer_traj)

        self.landmarks = torch.cat((self.preset_speaking[4], self.transfer_trajectory.view(-1, self.preset_speaking[4].shape[-1])), 0)

        self.transfer_diff = num_last.item()
        self.transfer_idx_start = self.transfer_diff*2
        self.idx_transfer_trajectory = torch.arange(self.transfer_idx_start, self.transfer_idx_start+len(self.preset[0])*NUM_TRANSFER_SAMPLE).view(len(self.preset[0]), NUM_TRANSFER_SAMPLE)


    def __call__(self, speaking):
        return self.generate_trajectory(speaking)

    def generate_trajectory(self, speaking):
        seek = 0
        now = self.start_idx + self.transfer_diff*(speaking[seek].eq(1))
        trajectory = [now.item()]
        while seek < len(speaking):
            speaking_now = speaking[max(seek-1, 0)]
            if self.keeping(speaking[seek:seek+SPEAKING_CHANGE_CHECK], speaking_now):
                next_node = self.sample_next_node(now)
                edge_trajectory = self.get_edge_trajectory(now, next_node)
                trajectory.extend(edge_trajectory.tolist())
                trajectory.append(next_node)

            else:
                if speaking_now == 0:
                    next_node = now + self.transfer_diff
                    transfer_trajectory = self.idx_transfer_trajectory[now]
                    trajectory.extend(transfer_trajectory.tolist())
                    trajectory.append(next_node)
                else:
                    breakpoint()
                    next_node = now - self.transfer_diff
                    transfer_trajectory = torch.flip(self.idx_transfer_trajectory[next_node], (0, ))
                    trajectory.extend(transfer_trajectory.tolist())
                    trajectory.append(next_node)

            seek = len(trajectory)-1
            now = next_node
        trajectory = trajectory[:len(speaking)]
        return trajectory

    def sample_next_node(self, now):
        is_speaking = now >= self.transfer_diff
        now_preset = self.preset_speaking if is_speaking else self.preset
        edges = now_preset[1]
        now_idx = now-self.transfer_diff if is_speaking else now

        node_candidate = [i for i in range(len(edges[now_idx])) if edges[now_idx][i]]
        next_node = sample(node_candidate, 1)[0]
        if is_speaking:
            next_node += self.transfer_diff

        return next_node

    def get_edge_trajectory(self, now, next_node):
        now_preset = self.preset if now < self.transfer_diff else self.preset_speaking
        idx_edges = now_preset[3]
        now_idx = now if now < self.transfer_diff else now-self.transfer_diff
        next_node_idx = next_node if now < self.transfer_diff else next_node-self.transfer_diff

        return idx_edges[now_idx, next_node_idx]


    def process_wav(self, path, verbose=False):
        wav, sr = self.get_wav(path)
        seconds = wav.shape[1] / sr
        num_frames = int(seconds * self.fps)
        if verbose:
            print(f"| Duration {seconds:.3f} seconds wav file")
            print(f"| The number of frames: {num_frames}")

        return wav, sr, num_frames, seconds

    @staticmethod
    def get_wav(path):
        wav, sr = torchaudio.load(path)
        return wav, sr

    @staticmethod
    def keeping(seq:torch.Tensor, now:int):
        return all(item==now for item in seq.tolist())

if __name__ == "__main__":
    is_speaking = torch.where(torch.linspace(0, 5.3, 125).sin().abs() < 0.5, 1, 0)

    # model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
    #                           model='silero_vad',
    #                           force_reload=True)

    # (get_speech_timestamps,
    # _, read_audio,
    # *_) = utils
    # wav, sr = torchaudio.load("piui.wav")
    # wav = torchaudio.functional.resample(wav, sr, 16000)
    # vad = get_speech_timestamps(wav.squeeze(), model, sampling_rate=16000)

    t = Trajectory(0)
    trajectory = t(is_speaking)
    breakpoint()
