import time
import torch
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

from wake_word_v1 import WakeWordCRNN

# ---------------------------------------
# Load the Model
# ---------------------------------------
model = WakeWordCRNN()

model.load_state_dict(
    torch.load(
        "wakeword_crnn.pt",
        map_location = "cpu"
    )
)

model.eval()

# --------------------------------
# Settings
# --------------------------------
SAMPLE_RATE = 16000

RECORD_SECONDS = 1.0
THRESHOLD = 0.5

# --------------------------------
# continuous Buffer
# --------------------------------
def record_audio():

    audio = sd.rec(
        16000,
        samplerate = SAMPLE_RATE,
        channels = 1,
        dtype = "float32"
    )

    sd.wait()

    return audio.flatten()

# ----------------------------------
# Preprocess Audio
# ----------------------------------
def preprocess_audio(audio):

    if len(audio) < 16000:
        audio = np.pad(
            audio, 
            (0, 16000 - len(audio))
        )
    else:
        audio = audio[:16000]

    mfcc = librosa.feature.mfcc(
        y = audio,
        sr = 16000,
        n_mfcc = 13,
        n_fft = 400,
        hop_length = 160
    )

    mfcc = mfcc.T

    mfcc = mfcc[np.newaxis, :, :]

    mfcc = torch.tensor(
        mfcc, 
        dtype = torch.float32
    )

    mfcc = mfcc.unsqueeze(0)

    return mfcc

# -----------------------------------
# Prediction
# -----------------------------------

def predict(audio):

    mfcc = preprocess_audio(audio)

    with torch.no_grad():

        score = model(mfcc)

    return score.item()

# -------------------------------------
# Test Block
# -------------------------------------
print("Listening for Kyro ...")

last_trigger = 0

try:
    
    while True:

        audio = record_audio()

        score = predict(audio)

        print(f"Score: {score:.3f}")

        if score > THRESHOLD:

            now = time.time()

            if now - last_trigger > 3:

                print("\nKYRO DETECTED\n")

                last_trigger = now

except KeyboardInterrupt:
    print("\nStopped Listening.\n")
