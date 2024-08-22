import torch
import torchaudio

class AudioProcessor:
    def __init__(self, sr=16000):
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                force_reload=False)
        (self.get_vad_dict, _, read_audio, *_) = utils

        self.model = model
        self.sr = sr

    def __call__(self, wav, fps=25):
        if isinstance(wav, str):
            wav = self.wav2tensor(wav)

        seq = self.get_binary_seq(wav, fps)
        return seq

    def wav2tensor(self, path):
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        wav = wav[0]
        return wav

    def get_binary_seq(self, wav, fps):
        vad_seq = torch.zeros(len(wav), dtype=torch.bool)

        vad_dicts = self.get_vad_dict(wav, self.model, sampling_rate=self.sr)
        for item in vad_dicts:
            start, end = item["start"], item["end"]
            vad_seq[start:end] = True

        offset = self.sr // fps
        seq = torch.zeros(int(len(wav) / self.sr * fps)+5, dtype=torch.bool)
        for idx, i in enumerate(range(0, len(vad_seq), offset)):
            if vad_seq[i:i+offset].sum() > offset // 2:
                seq[idx] = 1
        seq = seq[:idx+1]
        return seq


if __name__ == "__main__":
    a = AudioProcessor()
    seq = a("en_example.wav")
    breakpoint()
