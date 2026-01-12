import sys
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import threading
import wave
import os
import random
from datetime import datetime
from pathlib import Path
import time
import subprocess
from matplotlib.widgets import Button

# Configure matplotlib to support Chinese characters
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
    def __init__(self, audio_base_path):
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
        if num_to_pick > 0:
            self.current_set = random.sample(self.all_items, num_to_pick)
        self.current_index = 0
        return self.current_set

    def get_current_item(self):
        if not self.current_set: self.generate_new_set()
        return self.current_set[self.current_index] if self.current_index < len(self.current_set) else None

    def next_item(self):
        self.current_index += 1
        return self.get_current_item()

    def get_progress(self):
        if not self.current_set: return "0/0"
        return f"{self.current_index + 1} of {len(self.current_set)}"

class SimpleAudioVisualizerWithSAI:
    def __init__(self, chunk_size=512, sample_rate=16000, save_dir="recordings", audio_ref_dir=None):
        self.chunk_size, self.sample_rate = chunk_size, sample_rate
        self.running = False
        self.practice_set = PracticeSet(audio_base_path=audio_ref_dir)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.recorded_frames, self.is_recording = [], False
        self.reference_audio_playing = False
        self.p = pyaudio.PyAudio()
        self._setup_visualization()

    def _setup_visualization(self):
        self.fig = plt.figure(figsize=(10, 7), facecolor='#1a1a2e')
        self.ax_main = self.fig.add_subplot(111)
        self.ax_main.axis('off')
        
        # --- ADJUSTED STATUS TEXT POSITION (Moved Y from 0.05 to 0.22) ---
        self.status_text = self.ax_main.text(0.05, 0.22, 'Ready', transform=self.ax_main.transAxes,
            color='lime', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
        
        # Word Display
        item = self.practice_set.get_current_item()
        txt = f"[Tone {item['tone']}] {item['chinese']}\n{item['pinyin']}" if item else "Folder Empty"
        self.practice_text = self.ax_main.text(0.5, 0.55, txt, transform=self.ax_main.transAxes,
            color='cyan', fontsize=40, ha='center', va='center', weight='bold',
            bbox=dict(boxstyle='round,pad=1', facecolor='black', alpha=0.9, edgecolor='cyan', lw=3))
        
        # Progress
        self.progress_text = self.ax_main.text(0.95, 0.92, self.practice_set.get_progress(), 
            transform=self.ax_main.transAxes, color='yellow', ha='right',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

        # --- ADJUSTED BUTTONS ---
        # [left, bottom, width, height]
        self.ax_play = plt.axes([0.15, 0.08, 0.20, 0.07])
        self.play_btn = Button(self.ax_play, 'Play Reference', color='cyan')
        self.play_btn.label.set_color('black')
        self.play_btn.on_clicked(self.play_reference_audio)
        
        self.ax_rec = plt.axes([0.40, 0.08, 0.20, 0.07])
        self.rec_btn = Button(self.ax_rec, 'Record', color='lime')
        self.rec_btn.label.set_color('black')
        self.rec_btn.on_clicked(self.toggle_recording_and_save)
        
        self.ax_next = plt.axes([0.65, 0.08, 0.20, 0.07])
        self.next_btn = Button(self.ax_next, 'Next Item', color='orange')
        self.next_btn.label.set_color('black')
        self.next_btn.on_clicked(self.next_practice_item)

        # Ensure layout has enough space at the bottom
        plt.subplots_adjust(bottom=0.25)

    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording: self.recorded_frames.append(in_data)
        return (in_data, pyaudio.paContinue)

    def toggle_recording_and_save(self, event):
        if not self.is_recording:
            self.recorded_frames = []
            self.is_recording = True
            self.rec_btn.label.set_text('Stop & Save')
            self.status_text.set_text('Recording...')
            self.status_text.set_color('yellow')
            self.practice_text.set_color('yellow') 
            self.practice_text.get_bbox_patch().set_edgecolor('yellow')
        else:
            self.is_recording = False
            self.rec_btn.label.set_text('Record')
            self.practice_text.set_color('cyan')
            self.practice_text.get_bbox_patch().set_edgecolor('cyan')
            if self.recorded_frames: self.save_recording()
        self.fig.canvas.draw_idle()

    def save_recording(self):
        item = self.practice_set.get_current_item()
        ts = datetime.now().strftime("%H%M%S")
        fn = f"audio_{item['chinese']}_{ts}.wav"
        fp = os.path.join(self.save_dir, fn)
        try:
            with wave.open(fp, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.recorded_frames))
            self.status_text.set_text(f'Saved: {fn}')
            self.status_text.set_color('lime')
        except:
            self.status_text.set_text('Error Saving')
            self.status_text.set_color('red')
        self.fig.canvas.draw_idle()

    def play_reference_audio(self, event):
        if self.reference_audio_playing: return
        item = self.practice_set.get_current_item()
        if not item: return
        path = self.practice_set.audio_base_path / item['audio']
        if path.exists():
            threading.Thread(target=self._play_wav, args=(path,), daemon=True).start()

    def _play_wav(self, path):
        self.reference_audio_playing = True
        self.status_text.set_text('Playing Reference...')
        self.status_text.set_color('cyan')
        self.fig.canvas.draw_idle()
        try:
            with wave.open(str(path), 'rb') as wf:
                stream = self.p.open(format=self.p.get_format_from_width(wf.getsampwidth()),
                                   channels=wf.getnchannels(), rate=wf.getframerate(), output=True)
                data = wf.readframes(1024)
                while data and self.running:
                    stream.write(data)
                    data = wf.readframes(1024)
                stream.close()
        finally:
            self.reference_audio_playing = False
            self.status_text.set_text('Ready')
            self.status_text.set_color('lime')
            self.fig.canvas.draw_idle()

    def next_practice_item(self, event):
        item = self.practice_set.next_item()
        if item:
            self.practice_text.set_text(f"[Tone {item['tone']}] {item['chinese']}\n{item['pinyin']}")
            self.progress_text.set_text(self.practice_set.get_progress())
            self.status_text.set_text('Ready')
            self.status_text.set_color('lime')
        else:
            self.status_text.set_text('Session Complete!')
            self.status_text.set_color('magenta')
            threading.Timer(1.5, self._exit_and_launch).start()
        self.fig.canvas.draw_idle()

    def _exit_and_launch(self):
        self.stop()
        plt.close('all')
        os._exit(0)

    def start(self):
        self.running = True
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate,
                                 input=True, stream_callback=self.audio_callback)
        plt.show()

    def stop(self):
        self.running = False
        if hasattr(self, 'stream'): 
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

if __name__ == "__main__":
    # Your specified path
    user_path = Path(r"C:\Users\maruk\carfac-SAI\python\src\carfac\mandarin_audio_one_syllable")
    audio_ref_dir = str(user_path) if user_path.exists() else "mandarin_audio_one_syllable"

    app = SimpleAudioVisualizerWithSAI(audio_ref_dir=audio_ref_dir)
    try:
        app.start()
    except KeyboardInterrupt:
        app.stop()