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
from pathlib import Path
import time
import subprocess

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
    def __init__(self, audio_base_path="mandarin_audio_one_syllable"):
        self.audio_base_path = Path(audio_base_path)
        self.vocab_items = [
            {"id": 1,  "chinese": "天", "pinyin": "tiān", "tone": "1", "audio": "01_天_1.wav"},
            {"id": 2,  "chinese": "心", "pinyin": "xīn",  "tone": "1", "audio": "02_心_1.wav"},
            {"id": 3,  "chinese": "车", "pinyin": "chē",  "tone": "1", "audio": "03_车_1.wav"},
            {"id": 4,  "chinese": "学", "pinyin": "xué",  "tone": "2", "audio": "04_学_2.wav"},
            {"id": 5,  "chinese": "人", "pinyin": "rén",  "tone": "2", "audio": "05_人_2.wav"},
            {"id": 6,  "chinese": "白", "pinyin": "bái",  "tone": "2", "audio": "06_白_2.wav"},
            {"id": 7,  "chinese": "老", "pinyin": "lǎo",  "tone": "3", "audio": "07_老_3.wav"},
            {"id": 8,  "chinese": "火", "pinyin": "huǒ",  "tone": "3", "audio": "08_火_3.wav"},
            {"id": 9,  "chinese": "狗", "pinyin": "gǒu",  "tone": "3", "audio": "09_狗_3.wav"},
            {"id": 10, "chinese": "叫", "pinyin": "jiào", "tone": "4", "audio": "10_叫_4.wav"},
            {"id": 11, "chinese": "骂", "pinyin": "mà",   "tone": "4", "audio": "11_骂_4.wav"},
            {"id": 12, "chinese": "去", "pinyin": "qù",   "tone": "4", "audio": "12_去_4.wav"},
        ]
        self.all_items = [i for i in self.vocab_items if (self.audio_base_path / i['audio']).exists()]
        self.current_set = []
        self.current_index = 0

    def generate_new_set(self):
        num_to_pick = min(3, len(self.all_items))
        self.current_set = random.sample(self.all_items, num_to_pick)
        self.current_index = 0
        return self.current_set

    def get_current_item(self):
        if not self.current_set: self.generate_new_set()
        return self.current_set[self.current_index] if self.current_index < len(self.current_set) else None

    def next_item(self):
        self.current_index += 1
        return self.get_current_item()

    def get_progress_string(self):
        return f"{self.current_index + 1}/{len(self.current_set)}"

    def get_audio_path(self, item):
        if item and 'audio' in item:
            return self.audio_base_path / item['audio']
        return None

class SimpleAudioVisualizerWithSAI:
    def __init__(self, chunk_size=512, sample_rate=16000, save_dir="recordings", audio_ref_dir="reference"):
        self.chunk_size, self.sample_rate = chunk_size, sample_rate
        self.running = False
        self.practice_set = PracticeSet(audio_base_path=audio_ref_dir)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.recorded_frames, self.is_recording = [], False
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
            self.recorded_frames, self.is_recording = [], True
            self.btn_rec.label.set_text('Stop & Save')
            self.btn_rec.ax.set_facecolor('#ff4444')
            self.status_text.set_text('● RECORDING...')
            self.status_text.set_color('red')
        else:
            self.is_recording = False
            self.btn_rec.label.set_text('Start Recording')
            self.btn_rec.ax.set_facecolor('lime')
            self.save_recording()
        self.fig.canvas.draw_idle()

    def save_recording(self):
        if not self.recorded_frames: return
        item = self.practice_set.get_current_item()
        ts = datetime.now().strftime("%H%M%S")
        filename = f"rec_{item['chinese']}_{ts}.wav"
        path = os.path.join(self.save_dir, filename)
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.recorded_frames))
        self.status_text.set_text(f'✓ Saved: {filename}')
        self.status_text.set_color('lime')
        self.recorded_frames = []

    def play_reference_audio(self, event=None):
        item = self.practice_set.get_current_item()
        path = self.practice_set.get_audio_path(item)
        if path and path.exists():
            threading.Thread(target=self._play_wav, args=(path,), daemon=True).start()

    def _play_wav(self, path):
        self.status_text.set_text('Playing reference...')
        self.status_text.set_color('cyan')
        with wave.open(str(path), 'rb') as wf:
            stream = self.p.open(format=self.p.get_format_from_width(wf.getsampwidth()),
                               channels=wf.getnchannels(), rate=wf.getframerate(), output=True)
            data = wf.readframes(1024)
            while data and self.running:
                stream.write(data); data = wf.readframes(1024)
            stream.close()
        self.status_text.set_text('Ready'); self.status_text.set_color('yellow')
        self.fig.canvas.draw_idle()

    def next_practice_item(self, event=None):
        item = self.practice_set.next_item()
        if item:
            self._update_display(item)
            threading.Timer(0.3, self.play_reference_audio).start()
        else:
            self.practice_text.set_text("Set Complete!")
            self.status_text.set_text("✓ All items finished")
            self.status_text.set_color('lime')
            plt.close(self.fig)
            self._launch_next_script()
        self.fig.canvas.draw_idle()

    def _update_display(self, item):
        if item:
            txt = f"{item['chinese']} ({item['pinyin']}) - {self.practice_set.get_progress_string()}"
            self.practice_text.set_text(txt)
        else:
            self.practice_text.set_text("No Audio Files Found")

    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording: self.recorded_frames.append(in_data)
        return (in_data, pyaudio.paContinue)

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate,
            input=True, frames_per_buffer=self.chunk_size, stream_callback=self.audio_callback)
        self.running = True
        self._update_display(self.practice_set.get_current_item())
        # cache_frame_data=False fixed the warning
        self.animation = animation.FuncAnimation(self.fig, lambda i: [self.practice_text, self.status_text], 
                                                 interval=100, cache_frame_data=False)
        plt.show()

    def stop(self):
        self.running = False
        if hasattr(self, 'stream') and self.stream: self.stream.stop_stream(); self.stream.close()
        if hasattr(self, 'p') and self.p: self.p.terminate()

    def _launch_next_script(self):
        # Defines the target filename
        target_file = "audio_only_two_syllable.py"
        current_dir = Path(__file__).parent
        
        # 1. Look in the SAME folder
        next_script = current_dir / target_file
        
        # 2. If not found, look in the specific sibling folder "session_1_tone_recognition"
        # (This matches the structure implied by your old path)
        if not next_script.exists():
            next_script = current_dir.parent / "session_2_tone_production" / target_file

        # 3. Launch if found
        if next_script.exists():
            print(f"🚀 Launching next script: {next_script}")
            subprocess.Popen([sys.executable, str(next_script)])
        else:
            print(f"⚠️ Could not find next script: {target_file}")
            print(f"   Checked in: {current_dir}")

if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    audio_dir = script_dir / "mandarin_audio_one_syllable"
    if not audio_dir.exists():
        audio_dir = script_dir.parent / "mandarin_audio_one_syllable"
    
    visualizer = SimpleAudioVisualizerWithSAI(audio_ref_dir=str(audio_dir))
    try: visualizer.start()
    finally: visualizer.stop()