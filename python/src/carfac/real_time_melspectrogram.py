import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import queue
import librosa

# --- Configuration (Optimized for Tones) ---
FS = 22050
N_MELS = 128
FMIN = 50
FMAX = 1500
DISPLAY_TIME = 3      # seconds shown on screen
N_FFT = 2048
HOP_LENGTH = 512
NUM_FRAMES = int(DISPLAY_TIME * FS / HOP_LENGTH)


class LiveMelSpectrogram:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.mel_buffer = np.full((N_MELS, NUM_FRAMES), -80.0)
        self._setup_ui()

    def _setup_ui(self):
        self.fig, self.ax = plt.subplots(figsize=(16, 9), facecolor='#121212')
        # Fill the entire figure — no margins
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self.im = self.ax.imshow(
            self.mel_buffer,
            aspect='auto',
            origin='lower',
            cmap='magma',
            extent=[0, DISPLAY_TIME, FMIN, FMAX],
            vmin=-70,
            vmax=0,
            interpolation='bilinear',
        )

        # Strip everything: ticks, labels, spines, frame
        self.ax.set_axis_off()

        # Try to open the window maximized / fullscreen across backends
        try:
            mng = plt.get_current_fig_manager()
            try:
                mng.window.state('zoomed')           # TkAgg on Windows
            except Exception:
                try:
                    mng.window.showMaximized()       # Qt
                except Exception:
                    try:
                        mng.full_screen_toggle()     # generic fallback
                    except Exception:
                        pass
        except Exception:
            pass

    def _audio_callback(self, indata, frames, time, status):
        self.audio_queue.put(indata.copy())

    def update_plot(self, frame):
        while not self.audio_queue.empty():
            data = self.audio_queue.get().flatten()

            mel_spec = librosa.feature.melspectrogram(
                y=data, sr=FS, n_fft=N_FFT, hop_length=HOP_LENGTH,
                n_mels=N_MELS, fmin=FMIN, fmax=FMAX
            )
            log_mel = librosa.power_to_db(mel_spec, ref=1.0)

            new_frames = log_mel.shape[1]
            if new_frames > NUM_FRAMES:
                log_mel = log_mel[:, -NUM_FRAMES:]
                new_frames = NUM_FRAMES

            self.mel_buffer = np.roll(self.mel_buffer, -new_frames, axis=1)
            self.mel_buffer[:, -new_frames:] = log_mel
            self.im.set_array(self.mel_buffer)

        return [self.im]

    def start(self):
        with sd.InputStream(samplerate=FS, channels=1, callback=self._audio_callback):
            self.ani = animation.FuncAnimation(
                self.fig, self.update_plot, interval=30,
                blit=True, cache_frame_data=False
            )
            plt.show()


if __name__ == "__main__":
    LiveMelSpectrogram().start()