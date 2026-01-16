import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import sounddevice as sd
import soundfile as sf
import threading
import random
import time
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import sys
import subprocess
import numpy as np
import csv 

# Try to import pypinyin
try:
    from pypinyin import pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    print("⚠️ pypinyin not found. Installing it automatically...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypinyin"])
    try:
        from pypinyin import pinyin, Style
        HAS_PYPINYIN = True
        print("✅ Installation successful!")
    except ImportError:
        HAS_PYPINYIN = False
        print("❌ Automatic installation failed. Pinyin will be disabled.")

class ToneIntroductionQuiz:
    def __init__(self):
        # 1. FONT CONFIGURATION
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False 
        
        # 2. PATH FINDING & LOADING
        self.script_dir = Path(__file__).parent.resolve()
        
        items_one = self._load_from_folder('mandarin_audio_one_syllable')
        items_two = self._load_from_folder('mandarin_audio_two_syllable')

        if not items_one and not items_two:
            print("❌ No audio folders found. Exiting.")
            sys.exit()

        # 3. SELECT 3 FROM EACH
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

        # 4. GAME STATE
        self.current_item = None
        self.answered = False
        self.question_count = 0
        self.max_questions = len(self.vocab_items) 
        self.question_start_time = 0
        self.timer_started = False
        self.results = []
        self.session_start_time = datetime.now()
        self.is_playing = False

        # 5. SETUP UI
        self.fig = plt.figure(figsize=(6, 8))
        self.fig.patch.set_facecolor('white')
        self._setup_interface()
        
        if self.vocab_items:
            self.current_item = self.vocab_items[0]
            self._update_display()
        else:
            print("Error: No vocab items loaded.")

    def _find_folder(self, folder_name):
        path = self.script_dir / folder_name
        if path.exists(): return path
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
        files = sorted(list(folder_path.glob("*.wav")))
        
        for f in files:
            try:
                parts = f.stem.split('_') 
                if len(parts) >= 3:
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
                        "audio_path": f, 
                        "syllables": syllable_count
                    })
            except ValueError:
                continue
        return items

    def _setup_interface(self):
        self.ax = self.fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.ax.axis('off')
        
        self.progress_text = self.ax.text(0.5, 0.35, '', fontsize=12, ha='center', color='#7f8c8d')
        self.status_text = self.ax.text(0.5, 0.22, 'Click Play to start', fontsize=10, ha='center', color='#7f8c8d')

        self.instructions = self.ax.text(0.5, 0.32,
            "Each audio contains one or two tones. Identify them.\n"
            "Enter digits (e.g. '1' or '4' for single; '12' or '31' for pairs).",
            ha='center', va='top', fontsize=9, color='black')

        ax_input = plt.axes([0.3, 0.20, 0.4, 0.06])
        self.text_input = TextBox(ax_input, '', color='white', hovercolor='#f9f9f9')
        
        self.answer_text = self.ax.text(0.5, 0.14, '', fontsize=14, ha='center', weight='bold', color='#34495e')
        self.feedback_text = self.ax.text(0.5, 0.08, '', fontsize=16, ha='center', weight='bold')

        self.ax_play = plt.axes([0.35, 0.42, 0.3, 0.07])
        self.btn_play = Button(self.ax_play, 'Play Audio', color='#5B5FED', hovercolor='#4B4FDD')
        self.btn_play.label.set_color('white')
        self.btn_play.on_clicked(self.play_audio)

        self.ax_action = plt.axes([0.3, 0.02, 0.4, 0.06])
        self.btn_action = Button(self.ax_action, 'Check', color='#3498db', hovercolor='#2980b9')
        self.btn_action.label.set_color('white')
        self.btn_action.on_clicked(self._handle_action)

    def _update_display(self):
        self.answered = False
        self.timer_started = False
        self.question_start_time = 0
        
        self.btn_action.label.set_text('Check')
        self.btn_action.ax.set_facecolor('#3498db')
        
        self.text_input.set_val('')
        self.answer_text.set_text('')
        self.feedback_text.set_text('')
        self.status_text.set_text('Click Play to hear the word')
        self.status_text.set_color('#7f8c8d')
        self.progress_text.set_text(f"Question {self.question_count + 1}/{self.max_questions}")
        self.fig.canvas.draw_idle()

    def play_audio(self, event):
        if self.is_playing or not self.current_item: return
        
        def _thread_run():
            self.is_playing = True
            self.btn_play.label.set_text('...')
            self.fig.canvas.draw_idle()
            
            try:
                fpath = self.current_item['audio_path']
                data, sr = sf.read(str(fpath))
                sd.play(data, sr)
                sd.wait()
                
                self.question_start_time = time.time()
                self.timer_started = True
                
                self.status_text.set_text('Ready for answer')
                self.status_text.set_color('#27ae60')
            except Exception as e:
                print(f"Audio Error: {e}")
                self.status_text.set_text('Audio Error')
            
            self.is_playing = False
            self.btn_play.label.set_text('Play Audio')
            self.fig.canvas.draw_idle()

        threading.Thread(target=_thread_run, daemon=True).start()

    def _handle_action(self, event):
        if not self.answered:
            self._check_answer()
        else:
            self.next_word(event)

    def _check_answer(self):
        if not self.timer_started:
            self.status_text.set_text('⚠️ Listen first!')
            self.fig.canvas.draw_idle()
            return
            
        user_input = self.text_input.text.strip()
        if not user_input: return

        elapsed_time = time.time() - self.question_start_time
        correct_tone = str(self.current_item['tone'])
        is_correct = (user_input == correct_tone)
        self.answered = True

        # --- FIX: Match keys exactly with CSV fieldnames ---
        self.results.append({
            'question_idx': self.question_count + 1,
            'chinese': self.current_item['chinese'],
            'pinyin': self.current_item['pinyin'],     # Added
            'syllables': self.current_item['syllables'],
            'audio': self.current_item['audio_path'].name, # Added
            'correct_tone': correct_tone,
            'user_answer': user_input,                 # Renamed from user_input
            'is_correct': is_correct,                  # Renamed from correct
            'time_seconds': round(elapsed_time, 2),    # Renamed from time
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        # ---------------------------------------------------

        self.btn_action.label.set_text('Next Question')
        self.btn_action.ax.set_facecolor('#27ae60')

        self.answer_text.set_text(f"Your answer: {user_input}")
        if is_correct:
            self.feedback_text.set_text('CORRECT!')
            self.feedback_text.set_color('#27ae60')
        else:
            self.feedback_text.set_text(f'Wrong (Correct: {correct_tone})')
            self.feedback_text.set_color('#e74c3c')
            
        self.status_text.set_text(f"{self.current_item['chinese']} - {self.current_item['pinyin']}")
        self.status_text.set_color('black')
        self.fig.canvas.draw_idle()

    def next_word(self, event):
        self.question_count += 1
        if self.question_count >= self.max_questions:
            self._save_results_to_file()
            plt.close(self.fig)
        else:
            self.current_item = self.vocab_items[self.question_count]
            self._update_display()

    def _save_results_to_file(self):
        filename = "session1_audio_results.csv"
        filepath = self.script_dir / filename
        
        file_exists = filepath.exists()
        
        try:
            with open(filepath, mode='a', newline='', encoding='utf-8') as file:
                # --- FIX: Expanded fieldnames to match data ---
                writer = csv.DictWriter(file, fieldnames=[
                    'question_idx', 'chinese', 'pinyin', 'syllables', 'audio',
                    'correct_tone', 'user_answer', 'is_correct', 'time_seconds', 'timestamp'
                ])
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerows(self.results)
                
            print(f"Results appended to {filepath}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    def show(self):
        plt.show()

if __name__ == '__main__':
    app = ToneIntroductionQuiz()
    app.show()