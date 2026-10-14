import torch
import torch.nn as nn
import torch.nn.functional as F

class WakeWordCRNN(nn.Module):
    def __init__(self):

        super(WakeWordCRNN, self).__init__()

        # -------------------------
        # CNN BLOCK
        # -------------------------
        self.conv1 = nn.Conv2d( # 1st convolutional neural network layer
            in_channels=1,
            out_channels=16,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.bn1 = nn.BatchNorm2d(16) # Batch size, 16

        self.conv2 = nn.Conv2d( # 2nd convolutional neural network layer
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.bn2 = nn.BatchNorm2d(32) # Batch size, 32

        # Pool only along frequency axis (MFCC dimension)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))

        # After pooling, freq dimension becomes 13 -> 6 (floor)
        # GRU input size = channels * freq_out
        self.gru_input_size = 32 * 6

        # -------------------------
        # RNN BLOCK
        # -------------------------
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        # -------------------------
        # FINAL CLASSIFIER
        # -------------------------
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        """
        x shape: (batch, 1, 96, 13)
        """

        # CNN block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.pool(x)  # (batch, 32, 96, 6)

        # Reshape for GRU
        batch_size, channels, time_steps, freq_bins = x.size()

        # Move time dimension to axis 1
        x = x.permute(0, 2, 1, 3)  # (batch, time, channels, freq)

        # Flatten channels + freq into features
        x = x.reshape(batch_size, time_steps, channels * freq_bins)

        # GRU
        out, _ = self.gru(x)  # (batch, time, 64)

        # Take last time step
        last_out = out[:, -1, :]  # (batch, 64)

        # Final probability
        logits = self.fc(last_out)
        probs = torch.sigmoid(logits)

        return probs

# ---------------------------------------------------------
# TEST BLOCK — RUNS WHEN YOU EXECUTE THE SCRIPT DIRECTLY
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Initializing CRNN model...")
    model = WakeWordCRNN()

    print("Creating dummy MFCC input tensor...")
    dummy_input = torch.randn(1, 1, 96, 13)

    print("Running forward pass...")
    output = model(dummy_input)

    print("\nModel output probability:")
    print(output)