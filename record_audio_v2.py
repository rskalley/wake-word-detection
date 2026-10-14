import sounddevice as sd
import numpy as np
import wave
import time
import psutil
import statistics
import os

# Audio settings
SAMPLE_RATE = 16000 # Necessary for human speech.
CHANNELS = 1 # Mono.
BUFFER_SIZE = 256  # lower latency.
OUTPUT_FILE = r"C:\Users\b0031\OneDrive\Desktop\dataset\recording.wav" # Where the recording is saved and what it's saved as.


def evaluate_performance(intervals, cpu_usage): # Displays a continuous evaluation of the performance of the application during inference.
    """Compute and display jitter and CPU statistics."""
    if not intervals or not cpu_usage:
        print("No performance data collected.") # Error message.
        return

    mean_interval = statistics.mean(intervals)
    stdev_interval = statistics.stdev(intervals)
    min_interval = min(intervals)
    max_interval = max(intervals)

    mean_cpu = statistics.mean(cpu_usage)
    min_cpu = min(cpu_usage)
    max_cpu = max(cpu_usage)

    print("\n--- Performance Evaluation ---")
    print(f"Mean interval: {mean_interval:.2f} ms")
    print(f"Jitter (std dev): {stdev_interval:.2f} ms")
    print(f"Min interval: {min_interval:.2f} ms")
    print(f"Max interval: {max_interval:.2f} ms")
    print(f"Mean CPU: {mean_cpu:.1f}%")
    print(f"Min CPU: {min_cpu:.1f}%")
    print(f"Max CPU: {max_cpu:.1f}%")


def main(): # Handles the start/stop of the recording.
    print("Recording... Press Ctrl+C to stop.") 

    intervals = []
    cpu_usage = []

    # Prepare WAV file
    wav_file = wave.open(OUTPUT_FILE, "wb") 
    wav_file.setnchannels(CHANNELS)
    wav_file.setsampwidth(2)  # 16-bit PCM
    wav_file.setframerate(SAMPLE_RATE)

    try:
        last = time.time()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32',
            blocksize=BUFFER_SIZE,
            latency='low'
        ) as stream:

            while True:
                pcm_buffer, _ = stream.read(BUFFER_SIZE)

                # Timing interval
                now = time.time()
                interval = (now - last) * 1000  # ms
                intervals.append(interval)
                last = now

                # CPU usage
                cpu = psutil.cpu_percent(interval=None)
                cpu_usage.append(cpu)

                # Convert float32 into int16
                pcm_int16 = np.int16(pcm_buffer * 32767)
                wav_file.writeframes(pcm_int16.tobytes())

                print(f"Interval: {interval:.2f} ms | CPU: {cpu:.1f}%")

    except KeyboardInterrupt: # The escape command to end the recording and stop the program (ctrl + c)
        print("\nRecording stopped.")

    finally:
        wav_file.close()
        print(f"Saved to {OUTPUT_FILE}")
        
        # Directory Check
        print("Saving to: ")
        print(os.path.abspath(OUTPUT_FILE)) # Where the file is saved to.
        evaluate_performance(intervals, cpu_usage)


if __name__ == "__main__":
    main()