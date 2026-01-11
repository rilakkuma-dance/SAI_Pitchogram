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

# Try to import pypinyin
try:
    from pypinyin import pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False

class ToneSpectrogramQuiz:
    def __init__(self):
        # 1. FONT CONFIGURATION
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False 
        
        # 2. PATH FINDING
        self.audio_base_path = self._find_audio_folder()
        
        if not self.audio_base_path:
            print("❌ No audio folder selected. Exiting.")
            return

        print(f"✅ Using audio from: {self.audio_base_path}")
        
        self.vocab_items = self._scan_audio_folder()

        if not self.vocab_items:
            print("⚠️ Folder found, but no .mp3 files inside!")
            self.vocab_items = [{"id":0, "chinese":"Error", "pinyin":"-", "tone":"0", "audio": None}]

        # 3. GAME STATE
        self.current_item = None
        self.current_audio_y = None  # Store audio data for playback
        self.current_audio_sr = None # Store sample rate
        
        self.answered = False
        self.question_count = 0
        self.max_questions = 5
        self.question_start_time = None
        self.timer_started = False
        self.spectrogram_shown = False
        self.used_words = set()
        self.results = []
        self.session_start_time = datetime.now()

        # 4. SETUP UI
        self.fig = plt.figure(figsize=(8, 9)) 
        self.fig.patch.set_facecolor('white')
        self._setup_interface()
        self._select_random_item()

    def _find_audio_folder(self):
        script_dir = Path(__file__).parent.resolve()
        potential_path = script_dir / 'mandarin_audio_two_syllable'
        if potential_path.exists(): return potential_path
        potential_path = script_dir.parent / 'mandarin_audio_two_syllable'
        if potential_path.exists(): return potential_path

        print("⚠️ Could not auto-detect 'mandarin_mandarin_audio_two_syllable' folder.")
        root = tk.Tk()
        root.withdraw() 
        folder_selected = filedialog.askdirectory(title="Select the 'mandarin_audio_two_syllable' folder")
        root.destroy()
        if folder_selected: return Path(folder_selected)
        return None

    def _scan_audio_folder(self):
        items = []
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
        # Spectrogram Area
        self.ax_spec = self.fig.add_axes([0.15, 0.55, 0.7, 0.35])
        self.ax_spec.set_title("Mel Spectrogram", fontsize=10)
        self.ax_spec.set_xlabel("Time")
        self.ax_spec.set_ylabel("Frequency (Hz)")
        
        self.im_spec = self.ax_spec.imshow(np.zeros((128, 100)), aspect='auto', origin='lower', cmap='magma')
        self.ax_spec.axis('off') 

        # Text Areas
        self.ax_ui = self.fig.add_axes([0.1, 0.0, 0.8, 0.5])
        self.ax_ui.axis('off')
        
        self.progress_text = self.ax_ui.text(0.5, 0.95, '', fontsize=12, ha='center', color='#7f8c8d')
        
        # Status text (will show filename later)
        self.status_text = self.ax_ui.text(0.5, 0.85, 'Click "Show Spectrogram" to start', fontsize=11, ha='center', color='#7f8c8d')
        
        self.ax_ui.text(0.5, 0.60, 'Type tone numbers (e.g. 14)', fontsize=11, ha='center', color='#666666')

        # Input Box
        ax_input = plt.axes([0.3, 0.22, 0.4, 0.06])
        self.text_input = TextBox(ax_input, '', color='white', hovercolor='#f9f9f9')
        
        # Feedback Text
        self.answer_text = self.ax_ui.text(0.5, 0.30, '', fontsize=14, ha='center', weight='bold', color='#34495e')
        self.feedback_text = self.ax_ui.text(0.5, 0.20, '', fontsize=16, ha='center', weight='bold')

        # Buttons
        self.btn_show = Button(plt.axes([0.30, 0.40, 0.4, 0.07]), 'Show Spectrogram', color='#9b59b6', hovercolor='#8e44ad')
        self.btn_show.label.set_color('white')
        self.btn_show.on_clicked(self.show_spectrogram_action)

        self.btn_check = Button(plt.axes([0.15, 0.02, 0.3, 0.05]), 'Check', color='#3498db', hovercolor='#2980b9')
        self.btn_check.label.set_color('white')
        self.btn_check.on_clicked(self.check_answer_button)

        self.btn_next = Button(plt.axes([0.55, 0.02, 0.3, 0.05]), 'Next', color='#27ae60', hovercolor='#229954')
        self.btn_next.label.set_color('white')
        self.btn_next.on_clicked(self.next_word)

    def _select_random_item(self):
        available = [i for i in self.vocab_items if i['id'] not in self.used_words]
        if not available:
            self.used_words = set()
            available = self.vocab_items
            
        self.current_item = random.choice(available)
        self.used_words.add(self.current_item['id'])
        
        self.answered = False
        self.timer_started = False
        self.spectrogram_shown = False
        self.question_start_time = None
        self.current_audio_y = None
        
        # Clear UI
        self.text_input.set_val('')
        self.answer_text.set_text('')
        self.feedback_text.set_text('')
        self.ax_spec.axis('off')
        self.im_spec.set_data(np.zeros((128, 10))) 
        
        self.status_text.set_text('Click "Show Spectrogram" to reveal')
        self.status_text.set_color('#7f8c8d')
        self.progress_text.set_text(f"Question {self.question_count + 1}/{self.max_questions}")
        self.fig.canvas.draw_idle()

    def show_spectrogram_action(self, event):
        if self.spectrogram_shown or not self.current_item: return
        
        try:
            # Load Audio (and store it for playback later)
            fpath = self.audio_base_path / self.current_item['audio']
            
            # Load with librosa
            self.current_audio_y, self.current_audio_sr = librosa.load(str(fpath), sr=None)
            self.current_audio_y, _ = librosa.effects.trim(self.current_audio_y, top_db=20)
            
            # Generate Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(y=self.current_audio_y, sr=self.current_audio_sr, n_mels=128, fmin=50, fmax=4000)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Update Plot
            self.im_spec.set_data(log_mel_spec)
            self.im_spec.set_clim(vmin=log_mel_spec.min(), vmax=log_mel_spec.max())
            self.im_spec.set_extent([0, log_mel_spec.shape[1], 0, 4000]) 
            self.ax_spec.axis('on')
            self.ax_spec.set_aspect('auto')
            
            # Start Timer
            self.question_start_time = time.time()
            self.timer_started = True
            self.spectrogram_shown = True
            
            self.status_text.set_text('Study the pattern and guess the tones')
            self.status_text.set_color('#27ae60')
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error loading spectrogram: {e}")
            self.status_text.set_text("Error loading audio file")

    def check_answer_button(self, event):
        if not self.spectrogram_shown:
            self.status_text.set_text('⚠️ Click Show Spectrogram first!')
            self.fig.canvas.draw_idle()
            return
            
        user_input = self.text_input.text.strip().replace(' ', '')
        if not user_input or self.answered: return

        elapsed = time.time() - self.question_start_time
        correct_tone = self.current_item['tone']
        is_correct = (user_input == correct_tone)
        self.answered = True

        self.results.append({
            'word': self.current_item['chinese'],
            'pinyin': self.current_item['pinyin'],
            'tone': correct_tone,
            'user': user_input,
            'correct': is_correct,
            'time': elapsed,
            'file': self.current_item['audio']
        })

        self.answer_text.set_text(f"Your answer: {user_input}")
        
        if is_correct:
            self.feedback_text.set_text('CORRECT!')
            self.feedback_text.set_color('#27ae60')
        else:
            self.feedback_text.set_text(f'Wrong (Correct: {correct_tone})')
            self.feedback_text.set_color('#e74c3c')
            
        # 1. Show Text Info
        file_name = self.current_item['audio']
        word_info = f"{self.current_item['chinese']} - {self.current_item['pinyin']}"
        self.status_text.set_text(f"{word_info}\nFile: {file_name}\n(Playing Audio...)")
        self.status_text.set_color('black')
        self.fig.canvas.draw_idle()
        
        # 2. Play Audio Feedback automatically
        if self.current_audio_y is not None:
            threading.Thread(target=lambda: sd.play(self.current_audio_y, self.current_audio_sr), daemon=True).start()

    def _save_results(self):
        """Saves results to 'result' folder with same format as audio quiz"""
        result_dir = Path(__file__).parent / 'result'
        result_dir.mkdir(exist_ok=True)
        
        timestamp = self.session_start_time.strftime('%Y%m%d_%H%M%S')
        filename = f"spectrogram_quiz_results_{timestamp}.txt"
        save_path = result_dir / filename
        
        correct_count = sum(1 for r in self.results if r['correct'])
        total = len(self.results)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("SPECTROGRAM TONE QUIZ RESULTS\n")
            f.write("=============================\n")
            f.write(f"Date: {self.session_start_time}\n")
            f.write(f"Score: {correct_count}/{total}\n\n")
            
            for idx, r in enumerate(self.results):
                f.write(f"Q{idx+1}: {r['word']} ({r['pinyin']})\n")
                f.write(f"   Audio File: {r['file']}\n")
                f.write(f"   Correct Tone: {r['tone']} | Your Answer: {r['user']}\n")
                f.write(f"   Result: {'CORRECT' if r['correct'] else 'WRONG'}\n")
                f.write(f"   Time: {r['time']:.2f}s\n")
                f.write("-" * 30 + "\n")
                
        print(f"\n✅ Results saved to: {save_path}")

    def next_word(self, event):
        self.question_count += 1
        if self.question_count >= self.max_questions:
            print("\n" + "="*30)
            print("Spectrogram Quiz Complete!")
            self._save_results()
            print("="*30)
            plt.close(self.fig)
        else:
            self._select_random_item()

    def show(self):
        if self.audio_base_path:
            plt.show()

if __name__ == '__main__':
    app = ToneSpectrogramQuiz()
    app.show()