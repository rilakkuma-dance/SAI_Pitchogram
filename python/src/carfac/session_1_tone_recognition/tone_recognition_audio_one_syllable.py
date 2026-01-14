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

# Try to import pypinyin
try:
    from pypinyin import pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False

class ToneIntroductionQuiz:
    def __init__(self):
        # 1. FONT CONFIGURATION
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False 
        
        # 2. PATH FINDING
        self.audio_base_path = self._find_audio_folder()
        
        if not self.audio_base_path:
            print("❌ No audio folder selected. Exiting.")
            sys.exit()

        print(f"✅ Using audio from: {self.audio_base_path}")
        self.vocab_items = self._scan_audio_folder()

        if not self.vocab_items:
            print("⚠️ Folder found, but no .wav files inside!")
            self.vocab_items = [{"id":0, "chinese":"Error", "pinyin":"-", "tone":"0", "audio": None}]

        # 3. GAME STATE
        self.current_item = None
        self.answered = False
        self.question_count = 0
        self.max_questions = 5
        self.question_start_time = 0
        self.timer_started = False
        self.used_words = set()
        self.results = []
        self.session_start_time = datetime.now()
        self.is_playing = False

        # 4. SETUP UI
        self.fig = plt.figure(figsize=(6, 8))
        self.fig.patch.set_facecolor('white')
        self._setup_interface()
        self._select_random_item()

    def _find_audio_folder(self):
        script_dir = Path(__file__).parent.resolve()
        potential_path = script_dir / 'mandarin_audio_one_syllable'
        if potential_path.exists(): return potential_path
        potential_path = script_dir.parent / 'mandarin_audio_one_syllable'
        if potential_path.exists(): return potential_path

        root = tk.Tk()
        root.withdraw() 
        folder_selected = filedialog.askdirectory(title="Select the 'mandarin_audio_one_syllable' folder")
        root.destroy()
        if folder_selected: return Path(folder_selected)
        return None

    def _scan_audio_folder(self):
        items = []
        if not self.audio_base_path.exists(): return items
        
        files = sorted(list(self.audio_base_path.glob("*.wav")))
        for f in files:
            try:
                parts = f.stem.split('_') 
                if len(parts) >= 3:
                    item_id = int(parts[0])
                    word = parts[1]
                    tone = parts[2]
                    
                    if HAS_PYPINYIN:
                        py_list = pinyin(word, style=Style.TONE)
                        pinyin_text = "".join([x[0] for x in py_list])
                    else:
                        pinyin_text = "---"

                    items.append({
                        "id": item_id, "chinese": word, "pinyin": pinyin_text,
                        "tone": tone, "audio": f.name
                    })
            except ValueError:
                continue
        return items

    def _setup_interface(self):
        self.ax = self.fig.add_axes([0.1, 0.1, 0.8, 0.8])
        self.ax.axis('off')
        
        self.progress_text = self.ax.text(0.5, 0.35, '', fontsize=12, ha='center', color='#7f8c8d')
        
        self.status_text = self.ax.text(0.5, 0.28, 'Click Play to start', fontsize=10, ha='center', color='#7f8c8d')

        ax_input = plt.axes([0.3, 0.20, 0.4, 0.06])
        self.text_input = TextBox(ax_input, '', color='white', hovercolor='#f9f9f9')
        
        self.answer_text = self.ax.text(0.5, 0.14, '', fontsize=14, ha='center', weight='bold', color='#34495e')
        self.feedback_text = self.ax.text(0.5, 0.08, '', fontsize=16, ha='center', weight='bold')

        # Play Button
        self.ax_play = plt.axes([0.35, 0.42, 0.3, 0.07])
        self.btn_play = Button(self.ax_play, 'Play Audio', color='#5B5FED', hovercolor='#4B4FDD')
        self.btn_play.label.set_color('white')
        self.btn_play.on_clicked(self.play_audio)

        # Single Action Button (Check then Next)
        self.ax_action = plt.axes([0.3, 0.02, 0.4, 0.06])
        self.btn_action = Button(self.ax_action, 'Check', color='#3498db', hovercolor='#2980b9')
        self.btn_action.label.set_color('white')
        self.btn_action.on_clicked(self._handle_action)

    def _select_random_item(self):
        available = [i for i in self.vocab_items if i['id'] not in self.used_words]
        if not available:
            self.used_words = set() # Reset if we run out
            available = self.vocab_items
            
        self.current_item = random.choice(available)
        self.used_words.add(self.current_item['id'])
        
        self.answered = False
        self.timer_started = False
        self.question_start_time = 0
        
        # Reset Button appearance
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
        if self.is_playing or not self.current_item or self.current_item['audio'] is None: return
        
        def _thread_run():
            self.is_playing = True
            self.btn_play.label.set_text('...')
            self.fig.canvas.draw_idle()
            
            try:
                fpath = self.audio_base_path / self.current_item['audio']
                data, sr = sf.read(str(fpath))
                sd.play(data, sr)
                sd.wait()
                
                # Start Timer AFTER audio finishes
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

        # Stop Timer
        elapsed_time = time.time() - self.question_start_time

        correct_tone = str(self.current_item['tone'])
        is_correct = (user_input == correct_tone)
        self.answered = True

        # Store detailed results for the report
        self.results.append({
            'word': self.current_item['chinese'],
            'pinyin': self.current_item['pinyin'],
            'audio': self.current_item['audio'],
            'correct_tone': correct_tone,
            'user_input': user_input,
            'correct': is_correct,
            'time': elapsed_time
        })

        # Update button to "Next" state
        self.btn_action.label.set_text('Next Question')
        self.btn_action.ax.set_facecolor('#27ae60')

        self.answer_text.set_text(f"Your answer: {user_input}")
        if is_correct:
            self.feedback_text.set_text('CORRECT!')
            self.feedback_text.set_color('#27ae60')
        else:
            self.feedback_text.set_text(f'Wrong (Correct: Tone {correct_tone})')
            self.feedback_text.set_color('#e74c3c')
            
        self.status_text.set_text(f"{self.current_item['chinese']} - {self.current_item['pinyin']}")
        self.status_text.set_color('black')
        self.fig.canvas.draw_idle()

    def next_word(self, event):
        self.question_count += 1
        if self.question_count >= self.max_questions:
            self._save_results()
            plt.close(self.fig)
            self._launch_next_script()
        else:
            self._select_random_item()

    def _save_results(self):
        result_dir = Path(__file__).parent / 'result'
        result_dir.mkdir(exist_ok=True)
        
        ts_str = self.session_start_time.strftime('%Y%m%d_%H%M%S')
        save_path = result_dir / f"audio_quiz_results_{ts_str}.txt"
        
        total_score = sum(1 for r in self.results if r['correct'])
        header_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        
        lines = []
        lines.append("AUDIO TONE QUIZ RESULTS")
        lines.append("=======================")
        lines.append(f"Date: {header_date}")
        lines.append(f"Score: {total_score}/{self.max_questions}\n")
        
        for i, res in enumerate(self.results):
            status = "CORRECT" if res['correct'] else "WRONG"
            lines.append(f"Q{i+1}: {res['word']} ({res['pinyin']})")
            lines.append(f"   Audio File: {res['audio']}")
            lines.append(f"   Correct Tone: {res['correct_tone']} | Your Answer: {res['user_input']}")
            lines.append(f"   Result: {status}")
            lines.append(f"   Time: {res['time']:.2f}s")
            lines.append("-" * 30)
            
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        print(f"✅ Report generated: {save_path}")

    def _launch_next_script(self):
        # Adjusted path logic to be slightly more robust to different environments
        next_script = Path(r"C:\Users\maruk\carfac-SAI\python\src\carfac\session_1_tone_recognition\tone_recognition_audio_two_syllable.py")
        if not next_script.exists():
             # Fallback: try looking in the same folder
             next_script = Path(__file__).parent / "tone_recognition_audio_two_syllable.py"

        if next_script.exists():
            subprocess.Popen([sys.executable, str(next_script)])
        else:
            print(f"⚠️ Could not find next script at: {next_script}")

    def show(self):
        plt.show()

if __name__ == '__main__':
    app = ToneIntroductionQuiz()
    app.show()