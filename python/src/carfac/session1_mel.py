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
import csv # Added for CSV support

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
        
        self.script_dir = Path(__file__).parent.resolve()
        
        # 1. LOAD ITEMS FROM BOTH FOLDERS
        items_one = self._load_from_folder('mandarin_audio_two_syllable')
        items_two = self._load_from_folder('mandarin_audio_two_syllable')

        if not items_one and not items_two:
            print("❌ No audio folders found. Exiting.")
            sys.exit()

        # 2. SELECT 3 FROM EACH (Total 6)
        selected_one = []
        selected_two = []

        if len(items_one) >= 3:
            selected_one = random.sample(items_one, 3)
        else:
            selected_one = items_one 
            
        if len(items_two) >= 3:
            selected_two = random.sample(items_two, 3)
        else:
            selected_two = items_two 
            
        self.vocab_items = selected_one + selected_two
        random.shuffle(self.vocab_items)
        
        print(f"✅ Loaded {len(self.vocab_items)} questions ({len(selected_one)} from 1-syllable, {len(selected_two)} from 2-syllable).")

        # Game State
        self.current_item = None
        self.current_audio_y = None
        self.current_audio_sr = None
        self.answered = False
        self.question_count = 0
        self.max_questions = len(self.vocab_items) # Should be 6
        self.spectrogram_shown = False
        self.results = []
        self.session_start_time = datetime.now()
        
        # Timer state
        self.question_start_time = 0

        # UI Setup
        self.fig = plt.figure(figsize=(8, 9)) 
        self.fig.patch.set_facecolor('white')
        self._setup_interface()
        
        if self.vocab_items:
            self.current_item = self.vocab_items[0]
            self._update_display()
        else:
            print("Error: No vocab items loaded.")

    def _find_folder(self, folder_name):
        # Check current dir
        path = self.script_dir / folder_name
        if path.exists(): return path
        # Check parent dir
        path = self.script_dir.parent / folder_name
        if path.exists(): return path
        return None

    def _load_from_folder(self, folder_name):
        folder_path = self._find_folder(folder_name)
        items = []
        
        if not folder_path:
            print(f"⚠️ Warning: Could not find folder '{folder_name}'")
            return items

        print(f"📂 Scanning: {folder_path}")
        files = sorted(list(folder_path.glob("*.m4a")) + list(folder_path.glob("*.wav")))
        
        for f in files:
            try:
                parts = f.stem.split('_') 
                if len(parts) >= 3:
                    # Assumes format: ID_Word_Tone.wav
                    tone = parts[-1]
                    word = parts[-2]
                    
                    if HAS_PYPINYIN:
                        py_list = pinyin(word, style=Style.TONE)
                        pinyin_text = "".join([x[0] for x in py_list])
                    else:
                        pinyin_text = "---"

                    syllable_count = 2 if 'two' in folder_name else 1

                    items.append({
                        "id": f.name, 
                        "chinese": word, 
                        "pinyin": pinyin_text,
                        "tone": tone, 
                        "audio_path": f, # Store full path
                        "syllables": syllable_count
                    })
            except ValueError:
                continue
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

    def _update_display(self):
        # Reset State
        self.answered = False
        self.spectrogram_shown = False
        self.question_start_time = time.time()  # START TIMER
        
        # Reset UI
        self.answer_text.set_text('')
        self.feedback_text.set_text('') 
        self.btn_action.label.set_text('Show Melspectrogram')
        self.btn_action.ax.set_facecolor('#3498db')
        
        self.im_spec.set_data(np.zeros((128, 100))) 
        self.ax_spec.axis('off')
        
        self.text_input.set_val('')
        
        self.status_text.set_text('Click button to reveal spectrogram')
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
        fpath = self.current_item['audio_path']
        self.current_audio_y, self.current_audio_sr = librosa.load(str(fpath), sr=None)
        
        # Generate the Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=self.current_audio_y, 
            sr=self.current_audio_sr, 
            n_mels=128, 
            fmin=50, 
            fmax=1000  # Focused on tone range
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # --- CALCULATE TIME IN SECONDS ---
        duration_sec = len(self.current_audio_y) / self.current_audio_sr
        
        # Update the data
        self.im_spec.set_data(log_mel_spec)
        self.im_spec.set_clim(vmin=log_mel_spec.min(), vmax=log_mel_spec.max())
        
        # --- UPDATE EXTENT TO USE SECONDS ---
        # set_extent ([left, right, bottom, top])
        self.im_spec.set_extent([0, duration_sec, 0, 1000]) 
        
        # Ensure axis labels are visible and accurate
        self.ax_spec.axis('on')
        self.ax_spec.set_xlabel('Time (seconds)')
        self.ax_spec.set_ylabel('Frequency (Hz)')
        
        self.spectrogram_shown = True
        self.btn_action.label.set_text('Check Answer')
        self.btn_action.ax.set_facecolor('#e67e22') 
        self.status_text.set_text('Analyze the pitch contour!')
        self.fig.canvas.draw_idle()

    def _check_answer(self):
        user_input = self.text_input.text.strip()
        if not user_input: return

        # STOP TIMER
        elapsed_time = time.time() - self.question_start_time

        correct_tone = str(self.current_item['tone'])
        is_correct = (user_input == correct_tone)
        self.answered = True

        # Store detailed results (Matching CSV columns)
        self.results.append({
            'question_idx': self.question_count + 1,
            'chinese': self.current_item['chinese'],
            'pinyin': self.current_item['pinyin'],
            'syllables': self.current_item['syllables'],
            'audio': self.current_item['audio_path'].name,
            'correct_tone': correct_tone,
            'user_answer': user_input,
            'is_correct': is_correct,
            'time_seconds': round(elapsed_time, 2),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        self.answer_text.set_text(f"Correct Tone: {correct_tone} | Your Ans: {user_input}")
        
        if is_correct:
            self.feedback_text.set_text('CORRECT!')
            self.feedback_text.set_color('#27ae60')
        else:
            self.feedback_text.set_text('WRONG')
            self.feedback_text.set_color('#e74c3c')
            
        self.btn_action.label.set_text('Next Item')
        self.btn_action.ax.set_facecolor('#27ae60')
        
        self.status_text.set_text(f"Playing: {self.current_item['chinese']} ({self.current_item['pinyin']})")
        self.fig.canvas.draw_idle()
        
        threading.Thread(target=lambda: sd.play(self.current_audio_y, self.current_audio_sr), daemon=True).start()

    def _next_word(self):
        self.question_count += 1
        if self.question_count >= self.max_questions:
            # End of quiz - Save and Close
            self._save_results_to_file()
            plt.close(self.fig)
        else:
            self.current_item = self.vocab_items[self.question_count]
            self._update_display()

    def _save_results_to_file(self):
        # 1. Use a fixed filename
        filename = "session1_mel_results.csv"
        filepath = self.script_dir / filename
        
        # 2. Check if file exists so we only write headers once
        file_exists = filepath.exists()
        
        try:
            with open(filepath, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=[
                    'question_idx', 'chinese', 'pinyin', 'syllables', 'audio',
                    'correct_tone', 'user_answer', 'is_correct', 'time_seconds', 'timestamp'
                ])
                
                # Only write the top header row if the file is brand new
                if not file_exists:
                    writer.writeheader()
                
                writer.writerows(self.results)
                
            print(f"Results appended to {filepath}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    def show(self):
        plt.show()

if __name__ == '__main__':
    ToneSpectrogramQuiz().show()