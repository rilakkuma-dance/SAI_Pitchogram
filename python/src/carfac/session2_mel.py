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
import csv
from datetime import datetime
import time
from pathlib import Path
import librosa 
import subprocess

# Try to import pypinyin
try:
    from pypinyin import pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False

# Configure matplotlib for Chinese characters
def setup_chinese_font():
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
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=128, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = 50 
        self.fmax = 4000 

    def process_chunk(self, audio_chunk):
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
        y, sr = librosa.load(str(audio_path), sr=self.sample_rate)
        y, _ = librosa.effects.trim(y, top_db=20) 
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels, 
            fmin=self.fmin, fmax=self.fmax,
            hop_length=self.hop_length # Ensure hop_length matches live processor
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec

class VisualizationHandler:
    def __init__(self, n_freq_bins, spec_width):
        self.n_freq_bins = n_freq_bins
        self.spec_width = spec_width
        self.img = np.full((n_freq_bins, spec_width), -80.0)

class PracticeSet:
    def __init__(self, script_dir):
        self.script_dir = script_dir
        self.current_set = []
        self.current_index = 0
        
        # 1. Load from both folders
        self.items_one = self._scan_folder('mandarin_audio_one_syllable')
        self.items_two = self._scan_folder('mandarin_audio_two_syllable')
        
        if not self.items_one and not self.items_two:
            print("❌ No audio files found in either folder.")
            sys.exit()

        # 2. Generate the mixed set (3 + 3)
        self.generate_mixed_set()

    def _find_folder(self, folder_name):
        path = self.script_dir / folder_name
        if path.exists(): return path
        path = self.script_dir.parent / folder_name
        if path.exists(): return path
        return None

    def _scan_folder(self, folder_name):
        folder_path = self._find_folder(folder_name)
        items = []
        if not folder_path: return items
        
        syllables = 2 if 'two' in folder_name else 1
        
        for f in sorted(folder_path.glob("*.wav")):
            parts = f.stem.split('_')
            if len(parts) >= 3:
                word = parts[-2]
                tone = parts[-1]
                
                if HAS_PYPINYIN:
                    py_list = pinyin(word, style=Style.TONE)
                    py = "".join([x[0] for x in py_list])
                else:
                    py = "---"
                
                items.append({
                    "id": f.name,
                    "chinese": word,
                    "pinyin": py,
                    "tone": tone,
                    "audio_path": f,
                    "syllables": syllables
                })
        return items

    def generate_mixed_set(self):
        selected_one = []
        selected_two = []
        
        if len(self.items_one) >= 3:
            selected_one = random.sample(self.items_one, 3)
        else:
            selected_one = self.items_one
            
        if len(self.items_two) >= 3:
            selected_two = random.sample(self.items_two, 3)
        else:
            selected_two = self.items_two
            
        self.current_set = selected_one + selected_two
        random.shuffle(self.current_set)
        self.current_index = 0
        
        print(f"✅ Generated set with {len(self.current_set)} items ({len(selected_one)} single, {len(selected_two)} double).")

    def get_current_item(self):
        if self.current_index < len(self.current_set):
            return self.current_set[self.current_index]
        return None

    def next_item(self):
        self.current_index += 1
        return self.get_current_item()

    def get_progress_string(self):
        return f"{self.current_index + 1}/{len(self.current_set)}"

class SimpleAudioVisualizerWithSAI:
    def __init__(self, chunk_size=512, sample_rate=16000, sai_width=250): # Reduced width for 2s window
        self.chunk_size, self.sample_rate, self.sai_width = chunk_size, sample_rate, sai_width
        self.audio_queue = queue.Queue(maxsize=50)
        self.running = False
        self.script_dir = Path(__file__).parent.resolve()

        self.practice_set = PracticeSet(self.script_dir)
        self.processor = SpectrogramProcessor(sample_rate=sample_rate)
        self.vis = VisualizationHandler(self.processor.n_mels, sai_width)
        
        self.is_recording = False
        self.recorded_frames = []
        self.reference_audio_playing = False
        
        self.save_dir = self.script_dir / "mel_recording"
        self.save_dir.mkdir(exist_ok=True)
        
        self.results = [] 

        self._setup_visualization()

    def _setup_visualization(self):
        self.fig = plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(3, 2, height_ratios=[6, 1.5, 0.5])

        # === LEFT: Live Spectrogram ===
        self.ax_left = self.fig.add_subplot(gs[0, 0])
        # Fixed extent to match sai_width
        self.im_left = self.ax_left.imshow(self.vis.img, aspect='auto', origin='lower',
                                         extent=[0, self.sai_width, 50, 4000], cmap='magma', vmin=-80, vmax=0)
        self.ax_left.set_title("Your Audio (Live)", color='lime')
        self.ax_left.axis('off')

        # === RIGHT: Reference Spectrogram ===
        self.ax_right = self.fig.add_subplot(gs[0, 1])
        # We initialize with a blank fixed-width buffer
        fixed_ref_buffer = np.full((128, self.sai_width), -80.0)
        self.im_right = self.ax_right.imshow(fixed_ref_buffer, aspect='auto', origin='lower',
                                         extent=[0, self.sai_width, 50, 4000], cmap='magma', vmin=-80, vmax=0)
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
        path = self.save_dir / filename
        
        try:
            with wave.open(str(path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.recorded_frames))
            
            # Log Result
            self.results.append({
                'item_idx': self.practice_set.current_index + 1,
                'chinese': item['chinese'],
                'pinyin': item['pinyin'],
                'syllables': item['syllables'],
                'ref_audio': item['audio_path'].name,
                'mel_recording': filename,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            self.status_text.set_text(f"✓ Saved: {filename}")
            self.status_text.set_color('lime')
        except Exception as e:
            self.status_text.set_text(f"Error: {str(e)}")

    def play_reference_audio(self, event=None):
        item = self.practice_set.get_current_item()
        path = item['audio_path']
        
        if not path.exists():
            self.status_text.set_text(f"File not found: {path.name}")
            self.fig.canvas.draw_idle()
            return

        try:
            full_spec = self.processor.get_full_spectrogram(path)
            
            # === CENTER THE REFERENCE ===
            # Create a blank buffer matching Live Window Size (self.sai_width)
            display_buffer = np.full((128, self.sai_width), -80.0)
            
            # Calculate centering position
            spec_w = full_spec.shape[1]
            if spec_w < self.sai_width:
                start_col = (self.sai_width - spec_w) // 2
                display_buffer[:, start_col : start_col + spec_w] = full_spec
            else:
                # Crop if too long
                display_buffer = full_spec[:, :self.sai_width]

            # Update the plot
            self.im_right.set_data(display_buffer)
            self.im_right.set_clim(vmin=np.min(display_buffer), vmax=np.max(display_buffer))
            self.ax_right.set_title(f"Reference: {item['chinese']} ({item['pinyin']})", color='cyan')
            # ============================

        except Exception as e:
            print(f"❌ SPECTROGRAM ERROR: {e}")

        if not self.reference_audio_playing:
            threading.Thread(target=self._play_wav, args=(path,), daemon=True).start()
        
        self.fig.canvas.draw_idle()

    def _play_wav(self, path):
        self.reference_audio_playing = True
        self.status_text.set_text(f"🔊 Playing: {path.name}")
        self.status_text.set_color('cyan')
        try:
            with wave.open(str(path), 'rb') as wf:
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
            self.practice_info.set_text(f"{item['chinese']} ({item['pinyin']}) - {self.practice_set.get_progress_string()}")
            self.play_reference_audio()
        else:
            self.status_text.set_text("✓ Saving CSV...")
            self.status_text.set_color('lime')
            
            self._save_results_to_csv()
            plt.close(self.fig)

    def _save_results_to_csv(self):
        filename = "session2_mel_results.csv"
        filepath = self.script_dir / filename
        file_exists = filepath.exists()
        
        try:
            with open(filepath, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=[
                    'item_idx', 'chinese', 'pinyin', 'syllables', 
                    'ref_audio', 'mel_recording', 'timestamp'
                ])
                if not file_exists:
                    writer.writeheader()
                writer.writerows(self.results)
            print(f"✅ Session log saved to {filepath}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

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
            
            # Shift Left, Insert at Right
            self.vis.img[:, :-1] = self.vis.img[:, 1:]
            self.vis.img[:, -1] = col

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate,
                                input=True, stream_callback=self.audio_callback)
        self.running = True
        threading.Thread(target=self.process_loop, daemon=True).start()
        
        item = self.practice_set.get_current_item()
        self.practice_info.set_text(f"{item['chinese']} ({item['pinyin']}) - 1/6")
        
        self.ani = animation.FuncAnimation(self.fig, self.update_vis, interval=30, blit=True)
        plt.show()

if __name__ == "__main__":
    app = SimpleAudioVisualizerWithSAI()
    try: app.start()
    except KeyboardInterrupt: pass