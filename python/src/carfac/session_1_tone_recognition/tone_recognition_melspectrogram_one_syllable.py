import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import numpy as np
import librosa
import sounddevice as sd
import threading
import random
import time
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import sys
import subprocess

# Configure matplotlib for Chinese characters
try:
    from pypinyin import pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False

class ToneSpectrogramQuiz:
    def __init__(self):
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False 
        
        self.audio_base_path = self._find_audio_folder()
        if not self.audio_base_path:
            sys.exit()

        self.vocab_items = self._scan_audio_folder()
        
        # Game State
        self.current_item = None
        self.current_audio_y = None
        self.current_audio_sr = None
        self.answered = False
        self.question_count = 0
        self.max_questions = 5
        self.spectrogram_shown = False
        self.used_words = set()
        self.results = []
        self.session_start_time = datetime.now()

        # UI Setup
        self.fig = plt.figure(figsize=(8, 9)) 
        self.fig.patch.set_facecolor('white')
        self._setup_interface()
        self._select_random_item()

    def _find_audio_folder(self):
        script_dir = Path(__file__).parent.resolve()
        paths = [script_dir / 'mandarin_audio_one_syllable', script_dir.parent / 'mandarin_audio_one_syllable']
        for p in paths:
            if p.exists(): return p
        root = tk.Tk(); root.withdraw()
        folder = filedialog.askdirectory(title="Select audio folder"); root.destroy()
        return Path(folder) if folder else None

    def _scan_audio_folder(self):
        items = []
        for f in self.audio_base_path.glob("*.wav"):
            parts = f.stem.split('_') 
            if len(parts) >= 3:
                py = "".join([x[0] for x in pinyin(parts[1], style=Style.TONE)]) if HAS_PYPINYIN else "---"
                items.append({"id": int(parts[0]), "chinese": parts[1], "pinyin": py, "tone": parts[2], "audio": f.name})
        return items

    def _setup_interface(self):
        # Spectrogram Area
        self.ax_spec = self.fig.add_axes([0.15, 0.55, 0.7, 0.35])
        self.im_spec = self.ax_spec.imshow(np.zeros((128, 100)), aspect='auto', origin='lower', cmap='magma')
        self.ax_spec.axis('off') 

        # UI Area
        self.ax_ui = self.fig.add_axes([0.1, 0.05, 0.8, 0.45])
        self.ax_ui.axis('off')
        self.progress_text = self.ax_ui.text(0.5, 0.95, '', fontsize=12, ha='center', color='#7f8c8d')
        self.status_text = self.ax_ui.text(0.5, 0.85, '', fontsize=11, ha='center', color='#7f8c8d')

        # Input
        ax_input = plt.axes([0.3, 0.25, 0.4, 0.06])
        self.text_input = TextBox(ax_input, 'Tone:', color='white', hovercolor='#f9f9f9')
        
        # Feedback
        self.answer_text = self.ax_ui.text(0.5, 0.40, '', fontsize=14, ha='center', weight='bold')
        self.feedback_text = self.ax_ui.text(0.5, 0.32, '', fontsize=16, ha='center', weight='bold')

        # DUAL ACTION BUTTON
        self.ax_btn = plt.axes([0.30, 0.10, 0.4, 0.08])
        self.btn_action = Button(self.ax_btn, 'Show Melspectrogram', color='#3498db', hovercolor='#3498db')
        self.btn_action.label.set_color('white')
        self.btn_action.on_clicked(self._handle_button_click)

    def _select_random_item(self):
        available = [i for i in self.vocab_items if i['id'] not in self.used_words]
        self.current_item = random.choice(available or self.vocab_items)
        self.used_words.add(self.current_item['id'])
        
        self.answered = False
        self.is_playing = False
        self.timer_started = False
        
        # 1. RESET FEEDBACK LABELS (Delete previous results)
        self.answer_text.set_text('')
        self.feedback_text.set_text('Feedback') # Reset to default label
        self.feedback_text.set_color('#7f8c8d')  # Reset to gray
        
        # 2. Reset Button to start state
        self.btn_action.label.set_text('Play Loop')
        self.btn_action.ax.set_facecolor('#5B5FED')
        
        # 3. Clear Visuals and Input
        self.vis.img[:] = 0
        self.text_input.set_val('')
        
        self.status_text.set_text('Click Play Loop to start')
        self.progress_text.set_text(f"Question {self.question_count + 1}/{self.max_questions}")
        
        self.fig.canvas.draw_idle()
        
    def _handle_button_click(self, event):
        if not self.spectrogram_shown:
            self._show_spectrogram()
        elif not self.answered:
            self._check_answer()
        else:
            self._next_word()

    def _show_spectrogram(self):
        fpath = self.audio_base_path / self.current_item['audio']
        self.current_audio_y, self.current_audio_sr = librosa.load(str(fpath), sr=None)
        
        # 
        mel_spec = librosa.feature.melspectrogram(y=self.current_audio_y, sr=self.current_audio_sr, n_mels=128, fmin=50, fmax=4000)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        self.im_spec.set_data(log_mel_spec)
        self.im_spec.set_clim(vmin=log_mel_spec.min(), vmax=log_mel_spec.max())
        self.im_spec.set_extent([0, log_mel_spec.shape[1], 0, 4000]) 
        self.ax_spec.axis('on')
        self.spectrogram_shown = True
        
        self.btn_action.label.set_text('Check Answer')
        self.btn_action.ax.set_facecolor('#3498db')
        self.status_text.set_text('Look at the mel-spectorgram and guess the tone!')
        self.fig.canvas.draw_idle()

    def _check_answer(self):
        user_input = self.text_input.text.strip()
        if not user_input: return

        correct_tone = str(self.current_item['tone'])
        is_correct = (user_input == correct_tone)
        self.answered = True

        self.results.append({'word': self.current_item['chinese'], 'correct': is_correct})
        self.answer_text.set_text(f"Correct Tone: {correct_tone} | Your Ans: {user_input}")
        
        if is_correct:
            self.feedback_text.set_text('CORRECT!'); self.feedback_text.set_color('#27ae60')
        else:
            self.feedback_text.set_text('WRONG'); self.feedback_text.set_color('#e74c3c')
            
        self.btn_action.label.set_text('Next Item')
        self.btn_action.ax.set_facecolor('#27ae60')
        self.status_text.set_text(f"Playing: {self.current_item['chinese']} ({self.current_item['pinyin']})")
        self.fig.canvas.draw_idle()
        
        threading.Thread(target=lambda: sd.play(self.current_audio_y, self.current_audio_sr), daemon=True).start()

    def _next_word(self):
        self.question_count += 1
        if self.question_count >= self.max_questions:
            plt.close(self.fig)
            print("Quiz Complete!")
        else:
            self._select_random_item()

    def show(self):
        plt.show()

if __name__ == '__main__':
    ToneSpectrogramQuiz().show()