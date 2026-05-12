import sys
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.font_manager as fm
import threading
import queue
import wave
import csv
import random
from datetime import datetime
from pathlib import Path
import librosa

# --- Configuration (Optimized for Tones) ---
FS = 22050          
N_MELS = 128        
FMIN = 50           
FMAX = 1500         
DISPLAY_TIME = 3    
N_FFT = 2048
HOP_LENGTH = 512
NUM_FRAMES = int(DISPLAY_TIME * FS / HOP_LENGTH)

# Try to import pypinyin
try:
    from pypinyin import pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False

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

class PracticeSet:
    def __init__(self, script_dir):
        self.script_dir = script_dir
        self.current_set = []
        self.current_index = 0
        self.items_one = self._scan_folder('mandarin_audio')
        self.items_two = self._scan_folder('mandarin_audio')
        
        if not self.items_one and not self.items_two:
            print("❌ No audio files found.")
            sys.exit()
        self.generate_mixed_set()

    def _find_folder(self, folder_name):
        paths = [self.script_dir / folder_name, self.script_dir.parent / folder_name]
        for p in paths:
            if p.exists(): return p
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
                py = "".join([x[0] for x in pinyin(word, style=Style.TONE)]) if HAS_PYPINYIN else "---"
                items.append({"id": f.name, "chinese": word, "pinyin": py, "audio_path": f, "syllables": syllables})
        return items

    def generate_mixed_set(self):
        s1 = random.sample(self.items_one, min(3, len(self.items_one)))
        s2 = random.sample(self.items_two, min(3, len(self.items_two)))
        self.current_set = s1 + s2
        random.shuffle(self.current_set)
        self.current_index = 0

    def get_current_item(self):
        return self.current_set[self.current_index] if self.current_index < len(self.current_set) else None

    def next_item(self):
        self.current_index += 1
        return self.get_current_item()

class TonePracticeApp:
    def __init__(self):
        self.script_dir = Path(__file__).parent.resolve()
        self.practice_set = PracticeSet(self.script_dir)
        self.audio_queue = queue.Queue()
        self.mel_buffer = np.full((N_MELS, NUM_FRAMES), -80.0)
        self.is_recording = False
        self.recorded_frames = []
        self.results = []
        
        self.save_dir = self.script_dir / "mel_recordings"
        self.save_dir.mkdir(exist_ok=True)

        self._setup_ui()

    def _setup_ui(self):
        self.fig = plt.figure(figsize=(12, 7), facecolor='#121212')
        gs = self.fig.add_gridspec(3, 2, height_ratios=[6, 1, 0.5])

        # Left: Live Spectrogram
        self.ax_left = self.fig.add_subplot(gs[0, 0])
        self.im_left = self.ax_left.imshow(self.mel_buffer, aspect='auto', origin='lower',
                                          cmap='magma', extent=[0, DISPLAY_TIME, FMIN, FMAX], vmin=-70, vmax=0)
        self.ax_left.set_title("Your Voice (Live)", color='lime')
        self.ax_left.tick_params(colors='white')

        # Right: Reference
        self.ax_right = self.fig.add_subplot(gs[0, 1])
        # Initialize with same VMIN/VMAX as live view
        self.im_right = self.ax_right.imshow(np.full((N_MELS, NUM_FRAMES), -80.0), aspect='auto', 
                                            origin='lower', cmap='magma', extent=[0, DISPLAY_TIME, FMIN, FMAX],
                                            vmin=-60, vmax=-10)
        self.ax_right.set_title("Reference Pattern", color='cyan')
        self.ax_right.axis('off')

        # Text Area
        self.ax_text = self.fig.add_subplot(gs[1, :], facecolor='none')
        self.ax_text.axis('off')
        self.practice_info = self.ax_text.text(0.5, 0.7, "", ha='center', fontsize=16, color='white', weight='bold')
        self.status_text = self.ax_text.text(0.5, 0.1, "Ready", ha='center', color='yellow')

        # Buttons
        from matplotlib.widgets import Button
        self.btn_play = Button(plt.axes([0.2, 0.05, 0.15, 0.05]), 'Play Reference', color='#2c2c2c', hovercolor='#3d3d3d')
        self.btn_play.label.set_color('cyan')
        self.btn_play.on_clicked(self.play_reference)

        self.btn_rec = Button(plt.axes([0.4, 0.05, 0.2, 0.05]), 'Start Recording', color='#2c2c2c', hovercolor='#3d3d3d')
        self.btn_rec.label.set_color('lime')
        self.btn_rec.on_clicked(self.toggle_recording)

        self.btn_next = Button(plt.axes([0.65, 0.05, 0.15, 0.05]), 'Next Item', color='#2c2c2c', hovercolor='#3d3d3d')
        self.btn_next.label.set_color('orange')
        self.btn_next.on_clicked(self.next_item)

    def _audio_callback(self, indata, frames, time, status):
        self.audio_queue.put(indata.copy())
        if self.is_recording:
            self.recorded_frames.append(indata.copy())

    def update_plot(self, frame):
        while not self.audio_queue.empty():
            data = self.audio_queue.get().flatten()
            
            mel_spec = librosa.feature.melspectrogram(
                y=data, sr=FS, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                n_mels=N_MELS, fmin=FMIN, fmax=FMAX
            )
            # Power to DB
            log_mel = librosa.power_to_db(mel_spec, ref=1.0)
            
            new_frames = log_mel.shape[1]
            if new_frames > NUM_FRAMES:
                log_mel = log_mel[:, -NUM_FRAMES:]
                new_frames = NUM_FRAMES

            self.mel_buffer = np.roll(self.mel_buffer, -new_frames, axis=1)
            self.mel_buffer[:, -new_frames:] = log_mel
            self.im_left.set_array(self.mel_buffer)
            
        return [self.im_left]

    def toggle_recording(self, event=None):
        if not self.is_recording:
            self.recorded_frames = []
            self.is_recording = True
            self.btn_rec.label.set_text("Stop & Save")
            self.status_text.set_text("● RECORDING...")
            self.status_text.set_color('red')
        else:
            self.is_recording = False
            self.btn_rec.label.set_text("Start Recording")
            self.save_audio()

    def save_audio(self):
        if not self.recorded_frames: return
        item = self.practice_set.get_current_item()
        ts = datetime.now().strftime('%H%M%S')
        filename = f"rec_{item['chinese']}_{ts}.wav"
        path = self.save_dir / filename
        
        audio_data = np.concatenate(self.recorded_frames)
        audio_int = (audio_data * 32767).astype(np.int16)
        
        with wave.open(str(path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(FS)
            wf.writeframes(audio_int.tobytes())
            
        self.results.append({'chinese': item['chinese'], 'file': filename, 'time': datetime.now().isoformat()})
        self.status_text.set_text(f"✓ Saved {filename}")
        self.status_text.set_color('lime')

    def play_reference(self, event=None):
        item = self.practice_set.get_current_item()
        y, sr = librosa.load(item['audio_path'], sr=FS)
        
        # Calculate Reference Spectrogram
        ref_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                                                n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
        # Use a fixed reference for db conversion to ensure visibility
        ref_db = librosa.power_to_db(ref_mel, ref=1.0)
        
        # Center the reference in the 3s window
        ref_display = np.full((N_MELS, NUM_FRAMES), -80.0)
        w = ref_db.shape[1]
        if w < NUM_FRAMES:
            start = (NUM_FRAMES - w) // 2
            ref_display[:, start:start+w] = ref_db
        else:
            ref_display = ref_db[:, :NUM_FRAMES]
        
        # UPDATE IMAGE DATA
        self.im_right.set_array(ref_display)
        self.ax_right.set_title(f"Ref: {item['chinese']} ({item['pinyin']})", color='cyan')
        
        # Explicitly redraw the reference plot
        self.fig.canvas.draw_idle()
        
        # Play audio
        threading.Thread(target=lambda: sd.play(y, sr), daemon=True).start()

    def next_item(self, event=None):
        item = self.practice_set.next_item()
        if item:
            self.practice_info.set_text(f"{item['chinese']} ({item['pinyin']}) - {self.practice_set.current_index+1}/6")
            # Clear reference view for next item
            self.im_right.set_array(np.full((N_MELS, NUM_FRAMES), -80.0))
            self.status_text.set_text("Ready")
            self.status_text.set_color('yellow')
            self.fig.canvas.draw_idle()
        else:
            self.finish_session()

    def finish_session(self):
        csv_path = self.script_dir / 'session_results.csv'
        file_exists = csv_path.exists()
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['chinese', 'file', 'time'])
            if not file_exists: writer.writeheader()
            writer.writerows(self.results)
        plt.close()

    def start(self):
        item = self.practice_set.get_current_item()
        self.practice_info.set_text(f"{item['chinese']} ({item['pinyin']}) - 1/6")
        
        with sd.InputStream(samplerate=FS, channels=1, callback=self._audio_callback):
            self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=30, blit=True, cache_frame_data=False)
            plt.show()

if __name__ == "__main__":
    app = TonePracticeApp()
    app.start()