import matplotlib.pyplot as plt
import numpy as np
import librosa
import sounddevice as sd
from matplotlib.animation import FuncAnimation

# --- 設定 (声調の可視化に最適化) ---
FS = 22050          
N_MELS = 128        
FMIN = 50           
FMAX = 1500         
DISPLAY_TIME = 3    
N_FFT = 2048
HOP_LENGTH = 512

# バッファの初期化
num_frames = int(DISPLAY_TIME * FS / HOP_LENGTH)
mel_buffer = np.full((N_MELS, num_frames), -80.0)

class SimpleToneVisualizer:
    def __init__(self):
        plt.rcParams['font.sans-serif'] = ['Arial']
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.fig.patch.set_facecolor('white')
        
        self.img = self.ax.imshow(
            mel_buffer, 
            aspect='auto', 
            origin='lower', 
            cmap='magma', 
            extent=[0, DISPLAY_TIME, FMIN, FMAX],
            animated=True
        )
        
        self.ax.set_title("Real-time Mel-spectrogram", fontsize=14)
        self.ax.set_ylabel("Frequency (Hz)")
        self.ax.set_xlabel("Time (s)")

        self.audio_queue = []
        self.stream = sd.InputStream(
            samplerate=FS, 
            channels=1, 
            callback=self._audio_callback
        )

    def _audio_callback(self, indata, frames, time, status):
        self.audio_queue.append(indata.copy())

    def _update(self, frame):
        global mel_buffer
        if len(self.audio_queue) > 0:
            data = np.concatenate(self.audio_queue).flatten()
            self.audio_queue = []

            # メルスペクトログラムの計算
            mel_spec = librosa.feature.melspectrogram(
                y=data, sr=FS, n_fft=N_FFT, 
                hop_length=HOP_LENGTH, n_mels=N_MELS, 
                fmin=FMIN, fmax=FMAX
            )
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            
            # --- 修正ポイント: サイズの不一致を防ぐ ---
            new_frames = log_mel.shape[1]
            if new_frames > num_frames:
                log_mel = log_mel[:, -num_frames:] # バッファより長い場合は切り詰める
                new_frames = num_frames

            # バッファを左にシフトして新しいデータを追加
            mel_buffer = np.roll(mel_buffer, -new_frames, axis=1)
            mel_buffer[:, -new_frames:] = log_mel
            
            self.img.set_array(mel_buffer)
            self.img.set_clim(vmin=-70, vmax=0)
        
        return [self.img]

    def start(self):
        with self.stream:
            # cache_frame_data=False を追加して警告を抑制
            self.ani = FuncAnimation(
                self.fig, self._update, interval=30, blit=True, cache_frame_data=False
            )
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    visualizer = SimpleToneVisualizer()
    visualizer.start()