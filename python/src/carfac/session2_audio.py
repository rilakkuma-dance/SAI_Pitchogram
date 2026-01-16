import sys
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.font_manager as fm
import threading
import wave
import os
import random
import csv
from datetime import datetime
from pathlib import Path
import time
import subprocess

# Try to import pypinyin
try:
    from pypinyin import pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False

def setup_chinese_font():
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'Heiti TC', 'Arial Unicode MS']
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
        
        # Determine syllable count from folder name
        syllables = 2 if 'two' in folder_name else 1
        
        for f in sorted(folder_path.glob("*.wav")):
            parts = f.stem.split('_')
            if len(parts) >= 3:
                # Format: ID_Word_Tone.wav
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
        # Select 3 from each
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
    def __init__(self, chunk_size=512, sample_rate=16000):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.script_dir = Path(__file__).parent.resolve()
        
        # Setup Data
        self.practice_set = PracticeSet(self.script_dir)
        self.save_dir = self.script_dir / "audio_recording"
        self.save_dir.mkdir(exist_ok=True)
        
        self.running = False
        self.is_recording = False
        self.recorded_frames = []
        self.results = [] # Store session data for CSV
        
        self._setup_visualization()

    def _setup_visualization(self):
        self.fig = plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(3, 1, height_ratios=[6, 1.5, 0.5])
        self.ax_main = self.fig.add_subplot(gs[0])
        self.ax_main.axis('off')
        
        self.practice_text = self.ax_main.text(0.5, 0.5, "", transform=self.ax_main.transAxes,
            color='white', ha='center', va='center', fontsize=28, weight='bold')
        self.status_text = self.ax_main.text(0.5, 0.1, "Ready", transform=self.ax_main.transAxes,
            color='yellow', ha='center', fontsize=12)

        from matplotlib.widgets import Button
        self.ax_play = plt.axes([0.25, 0.05, 0.15, 0.04])
        self.btn_play = Button(self.ax_play, 'Play Reference', color='cyan', hovercolor='lightblue')
        self.btn_play.on_clicked(self.play_reference_audio)

        self.ax_rec = plt.axes([0.42, 0.05, 0.18, 0.04])
        self.btn_rec = Button(self.ax_rec, 'Start Recording', color='lime', hovercolor='green')
        self.btn_rec.on_clicked(self.toggle_recording)

        self.ax_next = plt.axes([0.62, 0.05, 0.15, 0.04])
        self.btn_next = Button(self.ax_next, 'Next Item', color='orange', hovercolor='yellow')
        self.btn_next.on_clicked(self.next_practice_item)

        self.fig.patch.set_facecolor('#121212')
        plt.subplots_adjust(bottom=0.15)

    def toggle_recording(self, event=None):
        if not self.is_recording:
            # Start Recording
            self.recorded_frames = []
            self.is_recording = True
            self.btn_rec.label.set_text('Stop & Save')
            self.btn_rec.ax.set_facecolor('#ff4444')
            self.status_text.set_text('● RECORDING...')
            self.status_text.set_color('red')
        else:
            # Stop Recording
            self.is_recording = False
            self.btn_rec.label.set_text('Start Recording')
            self.btn_rec.ax.set_facecolor('lime')
            self.save_recording()
        self.fig.canvas.draw_idle()

    def save_recording(self):
        if not self.recorded_frames: return
        item = self.practice_set.get_current_item()
        if not item: return

        # Generate Filename
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rec_{item['chinese']}_{ts}.wav"
        path = self.save_dir / filename
        
        # Save WAV
        try:
            with wave.open(str(path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.recorded_frames))
            
            # Log Result
            self.results.append({
                'item_idx': self.practice_set.current_index + 1,
                'chinese': item['chinese'],
                'pinyin': item['pinyin'],
                'syllables': item['syllables'],
                'ref_audio': item['audio_path'].name,
                'audio_recording': filename,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            self.status_text.set_text(f'✓ Saved: {filename}')
            self.status_text.set_color('lime')
        except Exception as e:
            print(f"Error saving wav: {e}")
        
        self.recorded_frames = []

    def play_reference_audio(self, event=None):
        item = self.practice_set.get_current_item()
        if item and item['audio_path'].exists():
            threading.Thread(target=self._play_wav, args=(item['audio_path'],), daemon=True).start()

    def _play_wav(self, path):
        self.status_text.set_text('Playing reference...')
        self.status_text.set_color('cyan')
        try:
            with wave.open(str(path), 'rb') as wf:
                stream = self.p.open(format=self.p.get_format_from_width(wf.getsampwidth()),
                                   channels=wf.getnchannels(), rate=wf.getframerate(), output=True)
                data = wf.readframes(1024)
                while data and self.running:
                    stream.write(data)
                    data = wf.readframes(1024)
                stream.stop_stream()
                stream.close()
        except Exception as e:
            print(f"Playback error: {e}")
        self.status_text.set_text('Ready')
        self.status_text.set_color('yellow')
        self.fig.canvas.draw_idle()

    def next_practice_item(self, event=None):
        # Before moving on, if they haven't recorded anything, we log a 'skipped' entry
        current = self.practice_set.get_current_item()
        if current:
            # Check if we already logged a recording for this item index
            already_logged = any(r['item_idx'] == self.practice_set.current_index + 1 for r in self.results)
            if not already_logged:
                self.results.append({
                    'item_idx': self.practice_set.current_index + 1,
                    'chinese': current['chinese'],
                    'pinyin': current['pinyin'],
                    'syllables': current['syllables'],
                    'ref_audio': current['audio_path'].name,
                    'audio_recording': "SKIPPED",
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

        item = self.practice_set.next_item()
        
        if item:
            self._update_display(item)
            threading.Timer(0.3, self.play_reference_audio).start()
        else:
            self.practice_text.set_text("Set Complete!")
            self.status_text.set_text("✓ Saving CSV...")
            self.status_text.set_color('lime')
            
            self._save_results_to_csv()
            plt.close(self.fig)
            
        self.fig.canvas.draw_idle()

    def _save_results_to_csv(self):
        filename = "session2_audio_results.csv"
        filepath = self.script_dir / filename
        file_exists = filepath.exists()
        
        try:
            with open(filepath, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=[
                    'item_idx', 'chinese', 'pinyin', 'syllables', 
                    'ref_audio', 'audio_recording', 'timestamp'
                ])
                if not file_exists:
                    writer.writeheader()
                writer.writerows(self.results)
            print(f"✅ Session log saved to {filepath}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    def _update_display(self, item):
        if item:
            txt = f"{item['chinese']} ({item['pinyin']}) - {self.practice_set.get_progress_string()}"
            self.practice_text.set_text(txt)
        else:
            self.practice_text.set_text("No Audio Files")

    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording: self.recorded_frames.append(in_data)
        return (in_data, pyaudio.paContinue)

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate,
            input=True, frames_per_buffer=self.chunk_size, stream_callback=self.audio_callback)
        self.running = True
        
        # Init Display
        self._update_display(self.practice_set.get_current_item())
        
        self.animation = animation.FuncAnimation(self.fig, lambda i: [self.practice_text, self.status_text], 
                                                 interval=100, cache_frame_data=False)
        plt.show()

    def stop(self):
        self.running = False
        if hasattr(self, 'stream') and self.stream: 
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p') and self.p: 
            self.p.terminate()

if __name__ == "__main__":
    visualizer = SimpleAudioVisualizerWithSAI()
    try: visualizer.start()
    except KeyboardInterrupt: pass
    finally: visualizer.stop()