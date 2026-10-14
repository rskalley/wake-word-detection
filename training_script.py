import torch, os, librosa
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np

from wake_word_v1 import WakeWordCRNN
from dataset_loader_v1 import WakeWordDataset

# Testing Block
print("Current Working Directory: ")
print(os.getcwd()) # Gets current working directory.

# Dataset Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # File Path.
DATASET_DIR = os.path.join(BASE_DIR, "dataset") # Dataset folder path.
UNSEEN_DIR = os.path.join(DATASET_DIR, "unseen_samples")
dataset = WakeWordDataset(DATASET_DIR) # Instantiates where the dataset is.

# Truth Checks below
print("BASE_DIR: ", BASE_DIR) # Where is the Base Directory
print("DATASET_DIR: ", DATASET_DIR) # Where is the Dataset Directory
print(os.path.exists(DATASET_DIR)) # Does the Dataset Directory path exist
print(os.path.exists(os.path.join(DATASET_DIR, "positive"))) # Does the Positive folder exist
print(os.path.exists(os.path.join(DATASET_DIR, "negative"))) # Does the negative folder exist

# Load dataset
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model
model = WakeWordCRNN() # Assign my WakeWordCRNN as the model.
criterion = nn.BCELoss() # Calculates the loss /epoch.
optimizer = optim.Adam(model.parameters(), lr=0.001) #Adjust the weights using the Adam Optimizer.

# Training loop
epochs = 10 # How many times the model sees the training data.

for epoch in range(epochs): # For every epoch in the epoch range(10)
    total_loss = 0

    for mfcc, label in train_loader:
        optimizer.zero_grad()

        output = model(mfcc)  # (batch, 1)
        loss = criterion(output.view(-1), label.view(-1)) 

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

def test_unseen_folder(model, folder_path):

    print("\nTesting Unseen Samples")
    print("-" * 50)

    model.eval()

    with torch.no_grad():

        for filename in os.listdir(folder_path):

            if not filename.endswith(".wav"):
                continue

            filepath = os.path.join(folder_path, filename)

            # Load Audio
            audio, sr = librosa.load(filepath, sr=16000)

            # Ensure exactly one second
            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)))

            else:
                audio = audio[:16000]

            # Create MFCC
            mfcc = librosa.feature.mfcc(
                y = audio, 
                sr = 16000, 
                n_mfcc = 13,
                n_fft = 400,
                hop_length = 160
            )

            mfcc = mfcc.T
            mfcc = mfcc[np.newaxis, :, :]

            #Convert to tensor
            mfcc = torch.tensor(
                mfcc, 
                dtype = torch.float32
            )

            # Add batch dimension
            mfcc = mfcc.unsqueeze(0)

            score = model(mfcc).item()#

            prediction = (
                "KYRO"
                if score > 0.5
                else "NOT KYRO"
            )

            print(
                f"{filename: <25} "
                f"Score: {score:.3f} "
                f"Prediction: {prediction}"
            )

correct = 0
total = 0

with torch.no_grad():
    for mfcc, label in train_loader:
        output = model(mfcc)
        pred = (output.view(-1) > 0.5).float()
        correct += (pred == label).sum().item()
        total += label.size(0)

print("Accuracy:", correct / total)

print("\nSample Predictions: ")

with torch.no_grad():
    mfcc, label = next(iter(train_loader)) # Takes a single batch from train_loader.
    output = model(mfcc) # Passes all 16 samples through the model.

    for i in range(min(10, len(label))):
        print(
            f"Label: {label[i].item():.0f} | "
            f"Prediction: {output[i].item():.3f}"
        )

torch.save(model.state_dict(), "wakeword_crnn.pt")

if os.path.exists(UNSEEN_DIR):
    test_unseen_folder(model, UNSEEN_DIR)
else:
    print("No unseen_samples folder found")