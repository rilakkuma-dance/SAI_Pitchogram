import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, TextBox
import subprocess
import sys
import numpy as np
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import threading
import random
import os
from datetime import datetime
import time
from scipy import signal
from scipy.fft import fft

class SpectrogramProcessor:
    """Mel spectrogram processor"""
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=128, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.window = signal.windows.hann(n_fft)
        self.mel_basis = self._create_mel_filterbank()
    
    def _create_mel_filterbank(self):
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        fmin, fmax = 0, self.sample_rate / 2
        mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        filterbank = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for i in range(self.n_mels):
            left, center, right = bin_points[i:i+3]
            for j in range(left, center):
                filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                filterbank[i, j] = (right - j) / (right - center)
        return filterbank
    
    def process_audio(self, audio_data):
        """Process entire audio file into spectrogram"""
        spec_width = (len(audio_data) - self.n_fft) // self.hop_length + 1
        spectrogram = np.zeros((self.n_mels, spec_width))
        
        for i in range(spec_width):
            start = i * self.hop_length
            end = start + self.n_fft
            if end > len(audio_data):
                break
            
            chunk = audio_data[start:end]
            windowed = chunk * self.window
            spectrum = np.abs(fft(windowed)[:self.n_fft // 2 + 1])
            spectrum = 20 * np.log10(spectrum + 1e-10)
            spec_column = self.mel_basis @ spectrum
            spectrogram[:, i] = spec_column
        
        return spectrogram

class ToneSpectrogramQuiz:
    def __init__(self, audio_base_path=None):
        if audio_base_path is None:
            script_dir = Path(__file__).parent.resolve()
            possible_paths = [
                script_dir / 'reference',
                script_dir.parent / 'reference',
                script_dir / 'carfac' / 'reference',
            ]
            
            audio_base_path = None
            for path in possible_paths:
                if path.exists():
                    audio_base_path = path
                    print(f"✓ Found audio path: {audio_base_path}")
                    break
            
            if audio_base_path is None:
                print(f"⚠️ Warning: Could not find reference audio folder!")
                audio_base_path = script_dir / 'reference'
        
        self.audio_base_path = Path(audio_base_path)
        self.sample_rate = 16000
        
        # Spectrogram processor
        self.spec_processor = SpectrogramProcessor(sample_rate=self.sample_rate)
        
        # Vocabulary items
        self.vocab_items = [
            {"id": 1, "chinese": "书", "pinyin": "shū", "tone": "1", "audio": "men/1_men.wav"},
            {"id": 2, "chinese": "女人", "pinyin": "nǚrén", "tone": "32", "audio": "women/2_women.wav"},
            {"id": 3, "chinese": "雄", "pinyin": "xióng", "tone": "2", "audio": "men/3_men.wav"},
            {"id": 4, "chinese": "去", "pinyin": "qù", "tone": "4", "audio": "men/4_men.wav"},
            {"id": 6, "chinese": "喜欢", "pinyin": "xǐhuān", "tone": "31", "audio": "women/6_women.wav"},
            {"id": 7, "chinese": "街道", "pinyin": "jiēdào", "tone": "14", "audio": "women/7_women.wav"},
            {"id": 8, "chinese": "熊猫", "pinyin": "xióngmāo", "tone": "21", "audio": "men/8_men.wav"},
            {"id": 9, "chinese": "书店", "pinyin": "shūdiàn", "tone": "14", "audio": "women/9_women.wav"},
            {"id": 10, "chinese": "去年", "pinyin": "qùnián", "tone": "42", "audio": "men/10_men.wav"},
            {"id": 11, "chinese": "中午", "pinyin": "zhōngwǔ", "tone": "13", "audio": "women/11_women.wav"},
            {"id": 12, "chinese": "椅子", "pinyin": "yǐzi", "tone": "35", "audio": "men/12_men.wav"},
            {"id": 13, "chinese": "学校", "pinyin": "xuéxiào", "tone": "24", "audio": "women/13_women.wav"},
            {"id": 14, "chinese": "医院", "pinyin": "yīyuàn", "tone": "14", "audio": "men/14_men.wav"},
            {"id": 15, "chinese": "游戏", "pinyin": "yóuxì", "tone": "24", "audio": "women/15_women.wav"},
            {"id": 16, "chinese": "她", "pinyin": "tā", "tone": "1", "audio": "men/16_men.wav"},
        ]
        
        if not self.audio_base_path.exists():
            print(f"⚠️ Warning: Audio path does not exist: {self.audio_base_path}")
        else:
            print(f"✓ Using audio path: {self.audio_base_path}")
        
        self.current_item = None
        self.current_spectrogram = None
        self.answered = False
        self.question_count = 0
        self.max_questions = 5
        
        self.question_start_time = None
        self.question_elapsed_time = 0
        self.spectrogram_shown = False

        # Store already used words
        self.used_words = set()
        
        self.results = []
        self.session_start_time = datetime.now()
        
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.patch.set_facecolor('white')
        
        self._setup_interface()
        self._select_random_item()
        
    def _setup_interface(self):
        # Create grid layout
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1, 3, 1], hspace=0.3)
        
        # Top section: Instructions
        ax_top = self.fig.add_subplot(gs[0])
        ax_top.axis('off')
        ax_top.text(0.5, 0.8, 'Learn Tones from Spectrogram', 
                   fontsize=20, ha='center', va='center', weight='bold')
        ax_top.text(0.5, 0.4, 'Look at the spectrogram pattern and identify the tones', 
                   fontsize=12, ha='center', va='center', color='#666666')
        
        # Progress counter
        self.progress_text = ax_top.text(0.95, 0.9, '', 
                   fontsize=12, ha='right', va='top', weight='bold', color='#7f8c8d')
        
        # Middle section: Spectrogram display
        self.ax_spec = self.fig.add_subplot(gs[1])
        self.ax_spec.set_facecolor('#1a1a2e')
        
        # Initialize with empty spectrogram
        empty_spec = np.zeros((128, 200))
        self.im_spec = self.ax_spec.imshow(
            empty_spec, aspect='auto', origin='lower',
            interpolation='bilinear', cmap='magma', vmin=-80, vmax=0
        )
        self.ax_spec.set_title('Spectrogram (Click "Show Spectrogram" to reveal)', 
                              color='cyan', fontsize=14, weight='bold')
        self.ax_spec.set_xlabel('Time', color='white', fontsize=10)
        self.ax_spec.set_ylabel('Frequency', color='white', fontsize=10)
        self.ax_spec.tick_params(colors='white')
        
        # Bottom section: Controls
        ax_bottom = self.fig.add_subplot(gs[2])
        ax_bottom.axis('off')
        
        # Status text
        self.status_text = ax_bottom.text(
            0.5, 0.85, 'Click "Show Spectrogram" to see the tone pattern',
            fontsize=11, ha='center', va='center', color='#7f8c8d',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8)
        )
        
        # Instruction
        ax_bottom.text(0.5, 0.60, 'Type the correct tones (e.g., "14" for tones 1+4)', 
                      fontsize=10, ha='center', va='center', color='#666666')
        
        # Text input box
        from matplotlib.widgets import TextBox
        ax_input = plt.axes([0.3, 0.1, 0.4, 0.05])
        self.text_input = TextBox(ax_input, '', initial='', 
                                  color='white', hovercolor='#f9f9f9')
        
        # Answer and feedback
        self.answer_text = ax_bottom.text(0.5, 0.15, '', 
                   fontsize=13, ha='center', va='center', weight='bold', color='#34495e')
        
        self.feedback_text = ax_bottom.text(0.5, 0.03, '', 
                   fontsize=16, ha='center', va='center', weight='bold')
        
        # Buttons
        from matplotlib.widgets import Button
        
        ax_show = plt.axes([0.20, 0.02, 0.18, 0.05])
        self.btn_show = Button(ax_show, 'Show Spectrogram', color='#5B5FED', hovercolor='#4B4FDD')
        self.btn_show.label.set_color('white')
        self.btn_show.on_clicked(self.show_spectrogram)
        
        ax_check = plt.axes([0.41, 0.02, 0.18, 0.05])
        self.btn_check = Button(ax_check, 'Check Answer', color='#3498db', hovercolor='#2980b9')
        self.btn_check.label.set_color('white')
        self.btn_check.on_clicked(self.check_answer_button)
        
        ax_next = plt.axes([0.62, 0.02, 0.18, 0.05])
        self.btn_next = Button(ax_next, 'Next Word', color='#27ae60', hovercolor='#229954')
        self.btn_next.label.set_color('white')
        self.btn_next.on_clicked(self.next_word)
        
        self._update_progress()
        
    def _update_progress(self):
        self.progress_text.set_text(f"Question {self.question_count + 1}/{self.max_questions}")
        self.fig.canvas.draw_idle()
        
    def _select_random_item(self):
        # added this to prevent duplication of words
        random_item = random.choice(self.vocab_items)
        while random_item['id'] in self.used_words:
            random_item = random.choice(self.vocab_items)
        self.current_item = random_item
        self.used_words.add(random_item['id'])
        self.current_spectrogram = None
        self.answered = False
        self.spectrogram_shown = False
        self.question_start_time = None
        
        # Clear display
        empty_spec = np.zeros((128, 200))
        self.im_spec.set_data(empty_spec)
        self.ax_spec.set_title('Spectrogram (Click "Show Spectrogram" to reveal)', 
                              color='cyan', fontsize=14, weight='bold')
        
        self.status_text.set_text('Click "Show Spectrogram" to see the tone pattern')
        self.status_text.set_color('#7f8c8d')
        self.answer_text.set_text('')
        self.feedback_text.set_text('')
        self.text_input.set_val('')
        
        self.fig.canvas.draw_idle()
        self._update_progress()
        
        print(f"\n{'='*60}")
        print(f"NEW WORD (Question {self.question_count + 1}/{self.max_questions})")
        print(f"Pinyin: {self.current_item['pinyin']}")
        print(f"Correct tone: {self.current_item['tone']}")
        print(f"{'='*60}")
        
    def show_spectrogram(self, event=None):
        """Load and display the spectrogram"""
        if self.spectrogram_shown:
            return
            
        audio_path = self.audio_base_path / self.current_item['audio']
        
        if not audio_path.exists():
            self.status_text.set_text(f'⚠️ Audio file not found')
            self.status_text.set_color('red')
            print(f"⚠️ Audio file not found: {audio_path}")
            return
        
        try:
            # Load and process audio
            audio_data, sr = sf.read(str(audio_path))
            if sr != self.sample_rate:
                # Simple resampling
                num_samples = int(len(audio_data) * self.sample_rate / sr)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), num_samples),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Generate spectrogram
            self.current_spectrogram = self.spec_processor.process_audio(audio_data)
            
            # Display spectrogram
            self.im_spec.set_data(self.current_spectrogram)
            self.im_spec.set_clim(vmin=np.min(self.current_spectrogram), 
                                  vmax=np.max(self.current_spectrogram))
            
            # Start timer
            if not self.spectrogram_shown:
                self.question_start_time = time.time()
                self.spectrogram_shown = True
            
            self.status_text.set_text('Study the pattern and identify the tones')
            self.status_text.set_color('#27ae60')
            
            self.fig.canvas.draw_idle()
            
            print(f"✓ Spectrogram displayed, timer started")
            
        except Exception as e:
            self.status_text.set_text(f'❌ Error loading spectrogram')
            self.status_text.set_color('red')
            print(f"❌ Error: {e}")
    
    def check_answer_button(self, event):
        text = self.text_input.text
        if not text.strip():
            self.status_text.set_text('⚠️ Please enter an answer first')
            self.status_text.set_color('orange')
            self.fig.canvas.draw_idle()
            return
        
        if not self.spectrogram_shown:
            self.status_text.set_text('⚠️ Please view the spectrogram first')
            self.status_text.set_color('orange')
            self.fig.canvas.draw_idle()
            return
        
        self.check_answer(text)
    
    def check_answer(self, text):
        if not self.current_item or self.answered:
            return
        
        if self.question_start_time is not None:
            self.question_elapsed_time = time.time() - self.question_start_time
        else:
            self.question_elapsed_time = 0
        
        user_answer = text.strip().replace(' ', '').replace(',', '').replace('-', '')
        
        if not user_answer:
            return
        
        correct_answer = self.current_item['tone'].replace(',', '').replace('-', '')
        
        print(f"\n{'─'*60}")
        print(f"ANSWER: User='{user_answer}' | Correct='{correct_answer}'")
        print(f"Time: {self.question_elapsed_time:.2f}s")
        print(f"{'─'*60}\n")
        
        self.answered = True
        is_correct = (user_answer == correct_answer)
        
        result = {
            'question_number': self.question_count + 1,
            'chinese': self.current_item['chinese'],
            'pinyin': self.current_item['pinyin'],
            'correct_tone': correct_answer,
            'user_answer': user_answer,
            'is_correct': is_correct,
            'time_seconds': round(self.question_elapsed_time, 2),
            'audio_file': self.current_item['audio']
        }
        self.results.append(result)
        
        self.answer_text.set_text(f"Your answer: {user_answer}")
        
        if is_correct:
            self.feedback_text.set_text('✓ CORRECT!')
            self.feedback_text.set_color('#27ae60')
            self.status_text.set_text(f'Correct! ({self.current_item["pinyin"]})')
            self.status_text.set_color('#27ae60')
            print("✓ CORRECT!")
        else:
            self.feedback_text.set_text(f'✗ INCORRECT (Correct: {correct_answer})')
            self.feedback_text.set_color('#e74c3c')
            self.status_text.set_text(f'Incorrect - Correct: {correct_answer} ({self.current_item["pinyin"]})')
            self.status_text.set_color('#e74c3c')
            print(f"✗ INCORRECT!")
        
        self.fig.canvas.draw_idle()
    
    def next_word(self, event):
        self.question_count += 1
        
        if self.question_count >= self.max_questions:
            self.finish_quiz()
        else:
            self._select_random_item()
    
    def finish_quiz(self, event=None):
        print(f"\n{'='*60}")
        print(f"QUIZ COMPLETED!")
        print(f"{'='*60}\n")
        
        self._save_results_to_file()
        self._start_practice()
    
    def _save_results_to_file(self):
        try:
            script_dir = Path(__file__).parent
            results_dir = script_dir / 'tone_quiz_results'
            results_dir.mkdir(exist_ok=True)
            
            timestamp = self.session_start_time.strftime('%Y%m%d_%H%M%S')
            filename = f"tone_quiz_spectrogram_{timestamp}.txt"
            filepath = results_dir / filename
            
            total_questions = len(self.results)
            correct_count = sum(1 for r in self.results if r['is_correct'])
            accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0
            total_time = sum(r['time_seconds'] for r in self.results)
            avg_time = total_time / total_questions if total_questions > 0 else 0
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("MANDARIN TONE QUIZ - SPECTROGRAM LEARNING\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"Session Start: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Session End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Questions: {total_questions}\n")
                f.write(f"Correct Answers: {correct_count}\n")
                f.write(f"Accuracy: {accuracy:.1f}%\n")
                f.write(f"Total Time: {total_time:.2f} seconds\n")
                f.write(f"Average Time: {avg_time:.2f} seconds\n")
                f.write(f"\nMethod: Spectrogram-based tone recognition\n")
                f.write("\n" + "="*70 + "\n\n")
                
                for result in self.results:
                    f.write(f"Question {result['question_number']}/{self.max_questions}\n")
                    f.write(f"{'-'*70}\n")
                    f.write(f"Chinese:       {result['chinese']}\n")
                    f.write(f"Pinyin:        {result['pinyin']}\n")
                    f.write(f"Correct Tone:  {result['correct_tone']}\n")
                    f.write(f"Your Answer:   {result['user_answer']}\n")
                    f.write(f"Result:        {'✓ CORRECT' if result['is_correct'] else '✗ INCORRECT'}\n")
                    f.write(f"Time Taken:    {result['time_seconds']} seconds\n")
                    f.write(f"Audio File:    {result['audio_file']}\n")
                    f.write("\n")
                
                f.write("="*70 + "\n")
                f.write("END OF RESULTS\n")
                f.write("="*70 + "\n")
            
            print(f"\n{'='*70}")
            print(f"✅ RESULTS SAVED")
            print(f"Location: {filepath}")
            print(f"Accuracy: {accuracy:.1f}% ({correct_count}/{total_questions})")
            print(f"Avg Time: {avg_time:.2f}s")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"\n❌ Error saving: {e}")
    
    def _start_practice(self):
        print("\n" + "="*60)
        print("STARTING MEL SPECTROGRAM PRACTICE")
        print("="*60 + "\n")
        
        sd.stop()
        plt.close(self.fig)
        
        script_dir = Path(__file__).parent
        possible_scripts = [
            script_dir / 'understanding_tone_practice.py',
            script_dir.parent / 'understanding_tone_practice.py',
        ]
        
        main_script = None
        for script_path in possible_scripts:
            if script_path.exists():
                main_script = script_path
                break
        
        if main_script:
            print(f"✓ Launching: {main_script.name}")
            subprocess.Popen([sys.executable, str(main_script)])
        else:
            print("⚠️ Main practice script not found")
    
    def show(self):
        plt.show()

if __name__ == '__main__':
    print("\n" + "="*60)
    print("MANDARIN TONE QUIZ - SPECTROGRAM LEARNING")
    print("="*60)
    print("Learn to recognize tones by studying spectrograms!")
    print(f"Script location: {Path(__file__).parent}")
    
    quiz = ToneSpectrogramQuiz()
    quiz.show()