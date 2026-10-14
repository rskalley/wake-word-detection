import torch
import torch.nn as nn
from torch.utils.data import Dataset
import librosa
import numpy as np
import os

class WakeWordDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.root_dir = root_dir

        # Positive samples
        pos_dir = os.path.join(root_dir, "positive")
        for f in os.listdir(pos_dir):
            self.samples.append((os.path.join(pos_dir, f), 1))

        # Negative samples
        neg_dir = os.path.join(root_dir, "negative")
        for f in os.listdir(neg_dir):
            self.samples.append((os.path.join(neg_dir, f), 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]

        # Load audio
        audio, sr = librosa.load(filepath, sr=16000)

        # Ensure exactly 1 second
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)))
        else:
            audio = audio[:16000]

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=16000,
            n_mfcc=13,
            n_fft=400,
            hop_length=160
        )

        # mfcc shape: (13, 96)
        mfcc = mfcc.T  # → (96, 13)

        # Add channel dimension for CNN
        mfcc = mfcc[np.newaxis, :, :]  # (1, 96, 13)

        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

if __name__ == "__main__":
    dataset = WakeWordDataset("dataset/")
    print("Total samples: ", len(dataset))

    mfcc, label = dataset[0]
    print("MFCC Shape: ", mfcc.shape)
    print("Label: ", label)