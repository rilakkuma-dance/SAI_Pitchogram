import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
CHUNK = 1024        # Number of audio samples per frame
RATE = 44100        # Sampling rate

# Open audio stream from microphone
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Set up matplotlib figure
fig, ax = plt.subplots()
spec_data = np.random.rand(100, CHUNK//2 + 1)  # ✅ fix: +1
img = ax.imshow(spec_data, aspect='auto', origin='lower',
                extent=[0, CHUNK//2 + 1, 0, 100], cmap='viridis')
ax.set_xlabel("Frequency Bin")
ax.set_ylabel("Time Frames")
ax.set_title("Real-Time Spectrogram")

# Update function
def update(frame):
    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    fft_vals = np.abs(np.fft.rfft(data)) / CHUNK
    spec_data[:-1] = spec_data[1:]  # scroll
    spec_data[-1] = fft_vals
    img.set_data(spec_data)
    return [img]

ani = animation.FuncAnimation(fig, update, interval=30, blit=True,
                              cache_frame_data=False)  # ✅ suppress warning
plt.show()

# Clean up on close
stream.stop_stream()
stream.close()
p.terminate()
