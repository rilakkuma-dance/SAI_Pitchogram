import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, TextBox
import matplotlib.animation as animation
import subprocess
import sys
import numpy as np
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import librosa
import threading
import random
import os
from datetime import datetime
import time

# JAX/CARFAC/SAI imports
try:
    sys.path.append('./jax')
    import jax
    import jax.numpy as jnp
    import carfac.jax.carfac as carfac
    from carfac.np.carfac import CarParams
    import sai
    JAX_AVAILABLE = True
except ImportError:
    print("Warning: JAX/CARFAC/SAI not found. Install required packages.")
    JAX_AVAILABLE = False
    sys.exit(1)

# Import modules
from modules.visualization_handler import VisualizationHandler, SAIParams

# Audio Processor
class AudioProcessor:
    def __init__(self, fs=16000):
        self.fs = fs
        if JAX_AVAILABLE:
            try:
                self.hypers, self.weights, self.state = carfac.design_and_init_carfac(
                    carfac.CarfacDesignParameters(fs=fs, n_ears=1)
                )
                self.n_channels = self.hypers.ears[0].car.n_ch
                self.run_segment_jit = jax.jit(carfac.run_segment, static_argnames=['hypers', 'open_loop'])
                self.use_carfac = True
            except Exception as e:
                self.use_carfac = False
                self.n_channels = 200
        else:
            self.use_carfac = False
            self.n_channels = 200

    def process_chunk(self, audio_chunk):
        if self.use_carfac:
            try:
                if len(audio_chunk.shape) == 1:
                    audio_input = audio_chunk.reshape(-1, 1)
                else:
                    audio_input = audio_chunk
                audio_jax = jnp.array(audio_input, dtype=jnp.float32)
                naps, _, self.state, _, _, _ = self.run_segment_jit(
                    audio_jax, self.hypers, self.weights, self.state, open_loop=False
                )
                return np.array(naps[:, :, 0]).T
            except Exception as e:
                pass

        # Fallback
        try:
            if isinstance(audio_chunk, np.ndarray):
                chunk = audio_chunk.flatten()
            else:
                chunk = np.array(audio_chunk).flatten()

            if chunk.size == 0:
                return np.zeros((self.n_channels, 0), dtype=np.float32)

            abs_chunk = np.abs(chunk)
            nap = np.tile(abs_chunk, (self.n_channels, 1)).astype(np.float32)
            channel_scales = np.linspace(1.0, 0.1, num=self.n_channels, dtype=np.float32)[:, None]
            nap = nap * channel_scales
            return nap
        except Exception as e:
            return np.zeros((self.n_channels, 0), dtype=np.float32)

# SAI Processor
class SAIProcessor:
    def __init__(self, sai_params):
        self.sai_params = sai_params
        if JAX_AVAILABLE:
            try:
                self.sai = sai.SAI(sai_params)
                self.use_sai = True
            except Exception as e:
                self.use_sai = False
        else:
            self.use_sai = False
    
    def RunSegment(self, nap_output):
        if self.use_sai:
            try:
                return self.sai.RunSegment(nap_output)
            except Exception as e:
                return self._simple_sai(nap_output)
        else:
            return self._simple_sai(nap_output)
    
    def _simple_sai(self, nap_output):
        sai_output = np.zeros((self.sai_params.num_channels, self.sai_params.sai_width))
        
        for ch in range(min(nap_output.shape[0], self.sai_params.num_channels)):
            if nap_output.shape[1] > 0:
                channel_data = nap_output[ch, :]
                for lag in range(min(len(channel_data), self.sai_params.sai_width)):
                    if len(channel_data) > lag:
                        start_idx = max(0, len(channel_data) - lag - 10)
                        end_idx = len(channel_data) - lag
                        if end_idx > start_idx:
                            sai_output[ch, lag] = np.mean(channel_data[start_idx:end_idx])
        
        return sai_output


class ToneIntroductionQuizWithSAI:
    def __init__(self, audio_base_path=None):
        # Auto-detect audio path
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
        self.chunk_size = 512
        self.is_playing = False
        
        # SAI setup
        self.processor = AudioProcessor(fs=self.sample_rate)
        self.n_channels = self.processor.n_channels
        
        self.sai_params = SAIParams(
            num_channels=self.n_channels,
            sai_width=400,
            future_lags=399,
            num_triggers_per_frame=2,
            trigger_window_width=self.chunk_size + 1,
            input_segment_width=self.chunk_size,
            channel_smoothing_scale=0.5
        )
        
        self.sai_processor = SAIProcessor(self.sai_params)
        self.vis = VisualizationHandler(self.sample_rate, self.sai_params)
        
        # Audio data for visualization
        self.audio_data = None
        self.current_position = 0
        self.total_samples = 0
        self.is_visualizing = False
        
        # All words from VocabList
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
        
        self.current_item = None
        self.answered = False
        self.question_count = 0
        self.max_questions = 5
        
        # Timer variables
        self.question_start_time = None
        self.question_elapsed_time = 0
        self.timer_started = False

        # Store already used words
        self.used_words = set()
        
        # Results storage
        self.results = []
        self.session_start_time = datetime.now()
        
        self.fig = plt.figure(figsize=(10, 10))
        self.fig.patch.set_facecolor('white')
        
        self._setup_interface()
        self._select_random_item()
        
    def _setup_interface(self):
        # Main container
        main_ax = self.fig.add_axes([0.1, 0.05, 0.8, 0.9])
        main_ax.set_xlim(0, 1)
        main_ax.set_ylim(0, 1)
        main_ax.axis('off')
        
        # Tone visualization area (smaller now)
        viz_ax = self.fig.add_axes([0.2, 0.84, 0.6, 0.10])
        viz_ax.set_xlim(0, 1)
        viz_ax.set_ylim(0, 4)
        viz_ax.axis('off')
        
        # SAI Visualization (large central area)
        self.ax_sai = self.fig.add_axes([0.15, 0.42, 0.7, 0.38])
        self.im_sai = self.ax_sai.imshow(
            self.vis.img, aspect='auto', origin='upper',
            interpolation='bilinear', extent=[0, 200, 0, 200]
        )
        self.ax_sai.axis('off')
        
        # SAI label
        self.sai_label = self.ax_sai.text(
            0.02, 0.02, 'Click Play to see SAI pattern',
            transform=self.ax_sai.transAxes,
            verticalalignment='bottom', fontsize=11,
            color='cyan', weight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8)
        )
        
        # Progress counter
        self.progress_text = main_ax.text(0.5, 0.82, '', 
                    fontsize=11, ha='center', va='top', weight='bold',
                    color='#7f8c8d')
        
        # Play button
        ax_play = plt.axes([0.35, 0.34, 0.3, 0.05])
        self.btn_play = Button(ax_play, '▶ Play & Show SAI', color='#5B5FED', hovercolor='#4B4FDD')
        self.btn_play.label.set_color('white')
        self.btn_play.label.set_weight('bold')
        self.btn_play.on_clicked(self.play_audio)
        
        # Status text
        self.status_text = main_ax.text(0.5, 0.30, 'Click Play to hear & see the word', 
                    fontsize=9, ha='center', va='center', color='#7f8c8d')
        
        # Instruction text
        main_ax.text(0.5, 0.25, 'Type the correct tones number', 
                    fontsize=10, ha='center', va='center', color='#666666')
        
        main_ax.text(0.5, 0.21, 'Example: tiānqì → 14', 
                    fontsize=9, ha='center', va='center', color='#666666')
        
        # Text input box
        ax_input = plt.axes([0.2, 0.17, 0.6, 0.05])
        self.text_input = TextBox(ax_input, '', initial='', 
                                 color='white', hovercolor='#f9f9f9')
        
        # Answer display text
        self.answer_text = main_ax.text(0.5, 0.12, '', 
                    fontsize=12, ha='center', va='center', weight='bold',
                    color='#34495e')
        
        # Feedback text
        self.feedback_text = main_ax.text(0.5, 0.08, '', 
                    fontsize=14, ha='center', va='center', weight='bold')
        
        # Check Answer button
        ax_check = plt.axes([0.15, 0.01, 0.3, 0.04])
        self.btn_check = Button(ax_check, 'Check Answer', color='#3498db', hovercolor='#2980b9')
        self.btn_check.label.set_color('white')
        self.btn_check.on_clicked(self.check_answer_button)
        
        # Next Word button
        ax_next = plt.axes([0.55, 0.01, 0.3, 0.04])
        self.btn_next = Button(ax_next, 'Next Word', color='#27ae60', hovercolor='#229954')
        self.btn_next.label.set_color('white')
        self.btn_next.on_clicked(self.next_word)
        
        self._update_progress()
    
    def _update_progress(self):
        """Update the progress counter"""
        self.progress_text.set_text(f"Question {self.question_count + 1}/{self.max_questions}")
        self.fig.canvas.draw_idle()
        
    def _select_random_item(self):
        """Select a random vocabulary item without replacement"""

        # added this to prevent duplication of words
        random_item = random.choice(self.vocab_items)
        while random_item['id'] in self.used_words:
            random_item = random.choice(self.vocab_items)
        self.current_item = random_item
        self.used_words.add(random_item['id'])
        self.answered = False
        self.timer_started = False
        self.question_start_time = None
        
        # Clear SAI
        self.vis.img[:] = 0
        self.im_sai.set_data(self.vis.img)
        
        self.status_text.set_text('Click Play to hear & see the word')
        self.status_text.set_color('#7f8c8d')
        self.answer_text.set_text('')
        self.feedback_text.set_text('')
        self.text_input.set_val('')
        self.sai_label.set_text('Click Play to see SAI pattern')
        
        self.fig.canvas.draw_idle()
        self._update_progress()
        
        print(f"\n{'='*60}")
        print(f"NEW WORD SELECTED (Question {self.question_count + 1}/{self.max_questions})")
        print(f"{'='*60}")
        print(f"Pinyin: {self.current_item['pinyin']}")
        print(f"Correct tone: {self.current_item['tone']}")
        print(f"{'='*60}")
        
    def _process_audio_for_sai(self, audio_data):
        """Process entire audio file and generate SAI visualization"""
        self.vis.img[:] = 0
        
        # Process audio in chunks
        total_frames = len(audio_data) // self.chunk_size
        remaining = len(audio_data) % self.chunk_size
        
        for i in range(total_frames):
            start = i * self.chunk_size
            end = start + self.chunk_size
            chunk = audio_data[start:end]
            
            nap_output = self.processor.process_chunk(chunk)
            sai_output = self.sai_processor.RunSegment(nap_output)
            
            self.vis.get_vowel_embedding(nap_output)
            self.vis.run_frame(sai_output)
            
            if self.vis.img.shape[1] > 1:
                self.vis.img[:, :-1] = self.vis.img[:, 1:]
                self.vis.draw_column(self.vis.img[:, -1])
        
        # Process remaining samples
        if remaining > 0:
            start = total_frames * self.chunk_size
            chunk = np.pad(audio_data[start:], (0, self.chunk_size - remaining), 'constant')
            
            nap_output = self.processor.process_chunk(chunk)
            sai_output = self.sai_processor.RunSegment(nap_output)
            
            self.vis.get_vowel_embedding(nap_output)
            self.vis.run_frame(sai_output)
            
            if self.vis.img.shape[1] > 1:
                self.vis.img[:, :-1] = self.vis.img[:, 1:]
                self.vis.draw_column(self.vis.img[:, -1])
        
        # Update display
        current_max = np.max(self.vis.img) if self.vis.img.size else 1
        self.im_sai.set_data(self.vis.img)
        self.im_sai.set_clim(vmin=0, vmax=max(1, min(255, current_max * 1.3)))
        self.fig.canvas.draw_idle()
    
    def play_audio(self, event):
        """Show SAI visualization"""
        if self.is_playing or not self.current_item:
            return
        
        def _play():
            self.is_playing = True
            self.btn_play.label.set_text('Processing...')
            self.status_text.set_text('🔊 Playing & generating SAI...')
            self.status_text.set_color('#3498db')
            self.fig.canvas.draw_idle()
            
            try:
                audio_path = self.audio_base_path / self.current_item['audio']
                
                if not audio_path.exists():
                    print(f"⚠️ Audio file not found: {audio_path}")
                    self.status_text.set_text(f"⚠️ Audio file not found")
                    self.status_text.set_color('red')
                    self.is_playing = False
                    self.btn_play.label.set_text('▶ Play & Show SAI')
                    self.fig.canvas.draw_idle()
                    return
                
                # Load audio
                audio_data, sr = librosa.load(str(audio_path), sr=self.sample_rate)
                
                print(f"\n🔊 PLAYING & VISUALIZING:")
                print(f"   File: {audio_path.name}")
                print(f"   Chinese: {self.current_item['chinese']}")
                print(f"   Pinyin: {self.current_item['pinyin']}")
                
                # Play audio in separate thread while processing SAI
                # sd.play(audio_data, sr)
                
                # Process for SAI
                self._process_audio_for_sai(audio_data)
                
                # Wait for playback to finish
                # sd.wait()
                
                # Start timer after playback
                self.question_start_time = time.time()
                self.timer_started = True
                
                self.status_text.set_text('✓ Ready for your answer')
                self.status_text.set_color('#27ae60')
                print(f"✓ Playback & visualization complete\n")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                self.status_text.set_text(f"❌ Error: {str(e)[:30]}")
                self.status_text.set_color('red')
            
            self.is_playing = False
            self.btn_play.label.set_text('▶ Play & Show SAI')
            self.fig.canvas.draw_idle()
        
        threading.Thread(target=_play, daemon=True).start()
    
    def check_answer_button(self, event):
        """Check the answer when button is clicked"""
        text = self.text_input.text
        if not text.strip():
            self.status_text.set_text('⚠️ Please enter an answer first')
            self.status_text.set_color('orange')
            self.fig.canvas.draw_idle()
            return
        
        if not self.timer_started:
            self.status_text.set_text('⚠️ Please click Play first')
            self.status_text.set_color('orange')
            self.fig.canvas.draw_idle()
            return
        
        self.check_answer(text)
    
    def check_answer(self, text):
        """Check if the user's answer is correct"""
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
        print(f"ANSWER SUBMITTED")
        print(f"{'─'*60}")
        print(f"User answer: '{user_answer}'")
        print(f"Correct answer: '{correct_answer}'")
        print(f"Time taken: {self.question_elapsed_time:.2f} seconds")
        
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
            self.status_text.set_text('Great job!')
            self.status_text.set_color('#27ae60')
            print("✓ CORRECT!")
        else:
            self.feedback_text.set_text(f'✗ INCORRECT (Correct: {correct_answer})')
            self.feedback_text.set_color('#e74c3c')
            self.status_text.set_text('Try again with the next one')
            self.status_text.set_color('#e74c3c')
            print(f"✗ INCORRECT! Correct answer: {correct_answer}")
        
        print(f"{'─'*60}\n")
        self.fig.canvas.draw_idle()
    
    def _save_results_to_file(self):
        """Save all results to a text file"""
        try:
            script_dir = Path(__file__).parent
            results_dir = script_dir / 'tone_quiz_results'
            results_dir.mkdir(exist_ok=True)
            
            timestamp = self.session_start_time.strftime('%Y%m%d_%H%M%S')
            filename = f"tone_quiz_sai_{timestamp}.txt"
            filepath = results_dir / filename
            
            total_questions = len(self.results)
            correct_count = sum(1 for r in self.results if r['is_correct'])
            accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0
            total_time = sum(r['time_seconds'] for r in self.results)
            avg_time = total_time / total_questions if total_questions > 0 else 0
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("MANDARIN TONE INTRODUCTION QUIZ WITH SAI - RESULTS\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"Session Start: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Session End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Questions: {total_questions}\n")
                f.write(f"Correct Answers: {correct_count}\n")
                f.write(f"Accuracy: {accuracy:.1f}%\n")
                f.write(f"Total Time: {total_time:.2f} seconds\n")
                f.write(f"Average Time per Question: {avg_time:.2f} seconds\n")
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
            print(f"✅ RESULTS SAVED TO FILE")
            print(f"{'='*70}")
            print(f"Filename: {filename}")
            print(f"Location: {filepath}")
            print(f"Accuracy: {accuracy:.1f}% ({correct_count}/{total_questions} correct)")
            print(f"Average Time: {avg_time:.2f} seconds per question")
            print(f"{'='*70}\n")
            
            return filepath
            
        except Exception as e:
            print(f"\n❌ Error saving results: {e}")
            return None
    
    def next_word(self, event):
        """Move to next word"""
        self.question_count += 1
        
        if self.question_count >= self.max_questions:
            print(f"\n{'='*60}")
            print(f"COMPLETED {self.max_questions} QUESTIONS!")
            print(f"{'='*60}\n")
            
            self._save_results_to_file()
            
            print("\n✓ Quiz completed! Close window to exit.")
            
        else:
            self._select_random_item()
    
    def show(self):
        plt.show()


if __name__ == '__main__':
    print("\n" + "="*60)
    print("MANDARIN TONE INTRODUCTION QUIZ WITH SAI (5 Questions)")
    print("="*60)
    print(f"Script location: {Path(__file__).parent}")
    
    intro = ToneIntroductionQuizWithSAI()
    intro.show()