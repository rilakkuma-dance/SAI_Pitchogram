import sys
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.font_manager as fm
import threading
import queue
import wave
import os
import random
from datetime import datetime
import time
from pathlib import Path
import librosa 
import tkinter as tk

# Try to import pydub for MP3 support
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Configure matplotlib for Chinese characters
def setup_chinese_font():
    """Setup matplotlib to display Chinese characters"""
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'Arial Unicode MS']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            return True
    return False

setup_chinese_font()

class SpectrogramProcessor:
    """Processes audio frames into Mel Spectrograms"""
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=128, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        # Frequency range focused on voice (Matches research parameters)
        self.fmin = 50 
        self.fmax = 4000 

    def process_chunk(self, audio_chunk):
        """Used for LIVE scrolling spectrogram (Left side)"""
        if len(audio_chunk) < self.n_fft:
            audio_chunk = np.pad(audio_chunk, (0, self.n_fft - len(audio_chunk)), mode='constant')
            
        melspec = librosa.feature.melspectrogram(
            y=audio_chunk, sr=self.sample_rate, n_fft=self.n_fft, 
            hop_length=self.hop_length, n_mels=self.n_mels,
            fmin=self.fmin, fmax=self.fmax
        )
        melspec_db = librosa.power_to_db(melspec, ref=np.max)
        return np.mean(melspec_db, axis=1) if melspec_db.shape[1] > 0 else np.zeros(self.n_mels)

    def get_full_spectrogram(self, audio_path):
        """Processes a full file for STATIC display (Right side)"""
        y, sr = librosa.load(str(audio_path), sr=self.sample_rate)
        y, _ = librosa.effects.trim(y, top_db=20) 
        
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels, 
            fmin=self.fmin, fmax=self.fmax
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec

class VisualizationHandler:
    def __init__(self, n_freq_bins, spec_width):
        self.n_freq_bins = n_freq_bins
        self.spec_width = spec_width
        self.img = np.full((n_freq_bins, spec_width), -80.0)

class PracticeSet:
    def __init__(self, audio_base_path="carfac/mandarin_audio_two_syllable"):
        self.audio_base_path = Path(audio_base_path)
        self.all_items = [
            {"id": 1,  "chinese": "中国", "pinyin": "zhōngguó",  "tone": "12", "audio": "01_中国_12.wav", "type": "word"},
            {"id": 2,  "chinese": "商店", "pinyin": "shāngdiàn", "tone": "14", "audio": "02_商店_14.wav", "type": "word"},
            {"id": 3,  "chinese": "明天", "pinyin": "míngtiān",  "tone": "21", "audio": "03_明天_21.wav", "type": "word"},
            {"id": 4,  "chinese": "牛奶", "pinyin": "niúnǎi",    "tone": "23", "audio": "04_牛奶_23.wav", "type": "word"},
            {"id": 5,  "chinese": "学校", "pinyin": "xuéxiào",   "tone": "24", "audio": "05_学校_24.wav", "type": "word"},
            {"id": 6,  "chinese": "老师", "pinyin": "lǎoshī",    "tone": "31", "audio": "06_老师_31.wav", "type": "word"},
            {"id": 7,  "chinese": "美国", "pinyin": "měiguó",    "tone": "32", "audio": "07_美国_32.wav", "type": "word"},
            {"id": 8,  "chinese": "面包", "pinyin": "miànbāo",   "tone": "41", "audio": "08_面包_41.wav", "type": "word"},
            {"id": 9,  "chinese": "问题", "pinyin": "wèntí",     "tone": "42", "audio": "09_问题_42.wav", "type": "word"},
            {"id": 10, "chinese": "电脑", "pinyin": "diànnǎo",   "tone": "43", "audio": "10_电脑_43.wav", "type": "word"},
        ]
        self.current_set = []
        self.current_index = 0
        self.max_questions = 3

    def generate_new_set(self):
        self.current_set = random.sample(self.all_items, self.max_questions)
        self.current_index = 0
        return self.current_set

    def get_current_item(self):
        if not self.current_set: self.generate_new_set()
        return self.current_set[self.current_index]

    def next_item(self):
        self.current_index += 1
        return self.get_current_item() if self.current_index < self.max_questions else None

class SimpleAudioVisualizerWithSAI:
    def __init__(self, chunk_size=512, sample_rate=16000, sai_width=400):
        self.chunk_size, self.sample_rate, self.sai_width = chunk_size, sample_rate, sai_width
        self.audio_queue = queue.Queue(maxsize=50)
        self.running = False

        self.practice_set = PracticeSet()
        self.processor = SpectrogramProcessor(sample_rate=sample_rate)
        self.vis = VisualizationHandler(self.processor.n_mels, sai_width)
        
        self.is_recording = False
        self.recorded_frames = []
        self.reference_audio_playing = False
        self.save_dir = "recordings"
        os.makedirs(self.save_dir, exist_ok=True)

        self._setup_visualization()

    def _setup_visualization(self):
        self.fig = plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(3, 2, height_ratios=[6, 1.5, 0.5])

        # Left: Live Spectrogram
        self.ax_left = self.fig.add_subplot(gs[0, 0])
        self.im_left = self.ax_left.imshow(self.vis.img, aspect='auto', origin='lower',
                                         extent=[self.sai_width, 0, 50, 4000], cmap='magma', vmin=-80, vmax=0)
        self.ax_left.set_title("Your Audio (Live)", color='lime')
        self.ax_left.axis('off')

        # Right: Reference Spectrogram
        self.ax_right = self.fig.add_subplot(gs[0, 1])
        self.im_right = self.ax_right.imshow(np.full((128, 100), -80.0), aspect='auto', origin='lower',
                                           extent=[0, 100, 50, 4000], cmap='magma', vmin=-80, vmax=0)
        self.ax_right.set_title("Reference Pattern", color='cyan')
        self.ax_right.axis('off')

        self.ax_text = self.fig.add_subplot(gs[1, :])
        self.ax_text.axis('off')
        self.practice_info = self.ax_text.text(0.5, 0.6, "", ha='center', fontsize=16, color='white', weight='bold')
        self.status_text = self.ax_text.text(0.5, 0.2, "Ready", ha='center', color='yellow')

        # Buttons
        from matplotlib.widgets import Button
        self.btn_play = Button(plt.axes([0.25, 0.05, 0.15, 0.04]), 'Play Reference', color='cyan')
        self.btn_play.on_clicked(self.play_reference_audio)

        self.btn_rec = Button(plt.axes([0.42, 0.05, 0.18, 0.04]), 'Start Recording', color='lime')
        self.btn_rec.on_clicked(self.toggle_recording)

        self.btn_next = Button(plt.axes([0.62, 0.05, 0.15, 0.04]), 'Next Item', color='orange')
        self.btn_next.on_clicked(self.next_practice_item)

        self.fig.patch.set_facecolor('#121212')

    def toggle_recording(self, event=None):
        if not self.is_recording:
            self.recorded_frames = [] 
            self.is_recording = True
            self.btn_rec.label.set_text("Stop & Save")
            self.ax_left.set_title("Recording...", color='red')
            self.status_text.set_text("● RECORDING IN PROGRESS...")
            self.status_text.set_color('red')
        else:
            self.is_recording = False
            self.btn_rec.label.set_text("Start Recording")
            self.ax_left.set_title("Your Audio (Live)", color='lime')
            self.save_recorded_audio()
        self.fig.canvas.draw_idle()

    def save_recorded_audio(self):
        if not self.recorded_frames:
            self.status_text.set_text("No audio captured.")
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        item = self.practice_set.get_current_item()
        filename = f"rec_{item['chinese']}_{timestamp}.wav"
        path = os.path.join(self.save_dir, filename)
        
        try:
            with wave.open(path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.recorded_frames))
            self.status_text.set_text(f"✓ Saved: {filename}")
            self.status_text.set_color('lime')
        except Exception as e:
            self.status_text.set_text(f"Error: {str(e)}")

    def play_reference_audio(self, event=None):
        item = self.practice_set.get_current_item()
        path = self.practice_set.audio_base_path / item['audio']
        
        if not path.exists():
            self.status_text.set_text(f"File not found: {item['audio']}")
            self.fig.canvas.draw_idle()
            return

        # 1. Process and show the Mel-spectrogram on the right
        try:
            full_spec = self.processor.get_full_spectrogram(path)
            self.im_right.set_data(full_spec)
            
            # Adjust the extent [left, right, bottom, top] to match the spectrogram width
            # and the frequency range (50Hz to 4000Hz)
            self.im_right.set_extent([0, full_spec.shape[1], 50, 4000])
            
            # Auto-scale color limits based on the file's intensity
            self.im_right.set_clim(vmin=np.min(full_spec), vmax=np.max(full_spec))
            
            self.ax_right.set_title(f"Reference: {item['chinese']} ({item['pinyin']})", color='cyan')
        except Exception as e:
            print(f"Error processing spectrogram: {e}")

        # 2. Play the audio in a background thread
        if not self.reference_audio_playing:
            threading.Thread(target=self._play_wav, args=(path,), daemon=True).start()
        
        # Refresh the canvas
        self.fig.canvas.draw_idle()

    def _play_wav(self, path):
        self.reference_audio_playing = True
        self.status_text.set_text(f"🔊 Playing: {path.name}")
        self.status_text.set_color('cyan')
        try:
            with wave.open(str(path), 'rb') as wf:
                # Use format/channels/rate from the WAV file for perfect playback
                stream = self.p.open(
                    format=self.p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )
                data = wf.readframes(self.chunk_size)
                while data and self.running:
                    stream.write(data)
                    data = wf.readframes(self.chunk_size)
                stream.close()
        finally:
            self.reference_audio_playing = False
            self.status_text.set_text("Ready")
            self.status_text.set_color('yellow')
            self.fig.canvas.draw_idle()

    def next_practice_item(self, event=None):
        item = self.practice_set.next_item()
        if item:
            self.practice_info.set_text(f"{item['chinese']} ({item['pinyin']}) - {self.practice_set.current_index + 1}/3")
            self.play_reference_audio()
        else:
            self.status_text.set_text("✓ Practice Set Complete (3/3)")
            self.status_text.set_color('lime')
            self.btn_next.set_active(False)

    def update_vis(self, frame):
        self.im_left.set_data(self.vis.img)
        return [self.im_left]

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_float = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        if self.is_recording: 
            self.recorded_frames.append(in_data)
        self.audio_queue.put(audio_float)
        return (None, pyaudio.paContinue)

    def process_loop(self):
        while self.running:
            chunk = self.audio_queue.get()
            col = self.processor.process_chunk(chunk)
            self.vis.img[:, 1:] = self.vis.img[:, :-1]
            self.vis.img[:, 0] = col

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate,
                                input=True, stream_callback=self.audio_callback)
        self.running = True
        threading.Thread(target=self.process_loop, daemon=True).start()
        item = self.practice_set.get_current_item()
        self.practice_info.set_text(f"{item['chinese']} ({item['pinyin']}) - 1/3")
        self.ani = animation.FuncAnimation(self.fig, self.update_vis, interval=30, blit=True)
        plt.show()

if __name__ == "__main__":
    # Update this to your absolute path for stability
    audio_path = r"C:\Users\maruk\carfac-SAI\python\src\carfac\mandarin_audio_two_syllable"
    
    # Initialize the app with the specific path
    app = SimpleAudioVisualizerWithSAI()
    app.practice_set = PracticeSet(audio_base_path=audio_path)
    app.start()