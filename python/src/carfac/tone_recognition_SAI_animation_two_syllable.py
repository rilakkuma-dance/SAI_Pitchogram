import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, TextBox
import sys
import numpy as np
from pathlib import Path
import sounddevice as sd
import librosa
import random
import os
import time
from datetime import datetime
import subprocess

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

# -----------------------------------------------------------
# Audio Processor Class (High Quality)
# -----------------------------------------------------------
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
            except Exception:
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
            except Exception:
                pass
        
        # Fallback
        if isinstance(audio_chunk, np.ndarray):
            chunk = audio_chunk.flatten()
        else:
            chunk = np.array(audio_chunk).flatten()
        if chunk.size == 0:
            return np.zeros((self.n_channels, 0), dtype=np.float32)
        abs_chunk = np.abs(chunk)
        nap = np.tile(abs_chunk, (self.n_channels, 1)).astype(np.float32)
        channel_scales = np.linspace(1.0, 0.1, num=self.n_channels, dtype=np.float32)[:, None]
        return nap * channel_scales

# -----------------------------------------------------------
# SAI Processor Class
# -----------------------------------------------------------
class SAIProcessor:
    def __init__(self, sai_params):
        self.sai_params = sai_params
        if JAX_AVAILABLE:
            try:
                self.sai = sai.SAI(sai_params)
                self.use_sai = True
            except Exception:
                self.use_sai = False
        else:
            self.use_sai = False
    
    def RunSegment(self, nap_output):
        if self.use_sai:
            try:
                return self.sai.RunSegment(nap_output)
            except Exception:
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

# -----------------------------------------------------------
# MAIN QUIZ CLASS (High Quality Animation)
# -----------------------------------------------------------
class ToneIntroductionQuizWithSAI:
    def __init__(self, audio_base_path=None):
        # 1. Setup Audio Path
        if audio_base_path is None:
            script_dir = Path(__file__).parent.resolve()
            
            possible_paths = [
                script_dir / 'mandarin_audio_two_syllable', 
                script_dir.parent / 'mandarin_audio_two_syllable',
                script_dir / 'carfac' # Fallback only
            ]
            
            audio_base_path = None
            for path in possible_paths:
                if path.exists():
                    audio_base_path = path
                    print(f"✓ Found audio path: {audio_base_path}")
                    break
            
            if audio_base_path is None:
                audio_base_path = script_dir / 'mandarin_audio_two_syllable'
        
        self.audio_base_path = Path(audio_base_path)
        self.sample_rate = 16000
        self.chunk_size = 512
        
        # 2. Control Flags
        self.is_playing = False
        
        # 3. SAI Setup
        self.processor = AudioProcessor(fs=self.sample_rate)
        self.n_channels = self.processor.n_channels
        self.sai_params = SAIParams(
            num_channels=self.n_channels, sai_width=400, future_lags=399,
            num_triggers_per_frame=2, trigger_window_width=self.chunk_size + 1,
            input_segment_width=self.chunk_size, channel_smoothing_scale=0.1
        )
        self.sai_processor = SAIProcessor(self.sai_params)
        self.vis = VisualizationHandler(self.sample_rate, self.sai_params)
        
        # 4. Audio Playback Variables
        self.audio_data = None
        self.current_frame_index = 0
        
        # 5. Quiz Data
        self.vocab_items = [
            {"id": 1,  "chinese": "中国", "pinyin": "zhōngguó",  "tone": "12", "audio": "01_中国_12.wav"},
            {"id": 2,  "chinese": "商店", "pinyin": "shāngdiàn", "tone": "14", "audio": "02_商店_14.wav"},
            {"id": 3,  "chinese": "明天", "pinyin": "míngtiān",  "tone": "21", "audio": "03_明天_21.wav"},
            {"id": 4,  "chinese": "牛奶", "pinyin": "niúnǎi",    "tone": "23", "audio": "04_牛奶_23.wav"},
            {"id": 5,  "chinese": "学校", "pinyin": "xuéxiào",   "tone": "24", "audio": "05_学校_24.wav"},
            {"id": 6,  "chinese": "老师", "pinyin": "lǎoshī",    "tone": "31", "audio": "06_老师_31.wav"},
            {"id": 7,  "chinese": "美国", "pinyin": "měiguó",    "tone": "32", "audio": "07_美国_32.wav"},
            {"id": 8,  "chinese": "面包", "pinyin": "miànbāo",   "tone": "41", "audio": "08_面包_41.wav"},
            {"id": 9,  "chinese": "问题", "pinyin": "wèntí",     "tone": "42", "audio": "09_问题_42.wav"},
            {"id": 10, "chinese": "电脑", "pinyin": "diànnǎo",   "tone": "43", "audio": "10_电脑_43.wav"},
        ]
        
        self.current_item = None
        self.answered = False
        self.question_count = 0
        self.max_questions = 5
        self.used_words = set()
        self.results = []
        self.session_start_time = datetime.now()
        
        # Timer variables
        self.question_start_time = None
        self.question_elapsed_time = 0
        self.timer_started = False
        
        self.fig = plt.figure(figsize=(10, 10))
        self.fig.patch.set_facecolor('white')
        
        self._setup_interface()
        
        if self.vocab_items:
            self._select_random_item()
        
    def _setup_interface(self):
        # 1. Main container - hidden axis for text placement
        self.ax_ui = self.fig.add_axes([0, 0, 1, 1])
        self.ax_ui.axis('off')

        # 2. Spectrogram/SAI Area (Top)
        # Positioned to leave room for the "Question X/X" header
        self.ax_sai = self.fig.add_axes([0.12, 0.58, 0.76, 0.32])
        self.im_sai = self.ax_sai.imshow(
            self.vis.img, aspect='auto', origin='upper', cmap='inferno',
            extent=[0, 400, 0, self.processor.n_channels]
        )
        # Note: In the image, the axes are visible but dark
        self.ax_sai.set_facecolor('black')
        self.ax_sai.tick_params(colors='#666666', labelsize=8)

        # 3. Header and Subtitles
        self.progress_text = self.ax_ui.text(0.5, 0.50, 'Question 1/5', 
                                            ha='center', fontsize=12, color='#7f8c8d')
        self.status_text = self.ax_ui.text(0.5, 0.46, 'Click Play Loop to start', 
                                          ha='center', fontsize=10, color='#7f8c8d')

        # 5. Tone Input Box
        # Matches the long white box in the image
        self.ax_ui.text(0.28, 0.33, 'Tone:', ha='right', va='center', fontsize=10)
        ax_input = plt.axes([0.3, 0.30, 0.4, 0.06]) 
        self.text_input = TextBox(ax_input, '', color='white', hovercolor='#f9f9f9')

        # 6. Feedback Row
        self.ax_ui.text(0.35, 0.25, 'Your answer:', ha='right', fontsize=10, color='#7f8c8d')
        self.answer_text = self.ax_ui.text(0.36, 0.25, '', ha='left', fontsize=10, weight='bold')
        self.feedback_text = self.ax_ui.text(0.65, 0.25, 'Feedback', ha='left', fontsize=10, color='#7f8c8d')

        # 7. Big Play Button (Bottom)
        self.ax_btn = plt.axes([0.3, 0.12, 0.4, 0.08])
        self.btn_action = Button(self.ax_btn, 'Play Loop', color='#3498db', hovercolor='#3498db')
        self.btn_action.label.set_color('white')
        self.btn_action.label.set_weight('bold')
        self.btn_action.label.set_fontsize(14)
        self.btn_action.on_clicked(self._handle_button_click)

    def _check_answer(self):
        user_input = self.text_input.text.strip()
        if not user_input: return

        correct_tone = str(self.current_item['tone'])
        is_correct = (user_input == correct_tone)
        self.answered = True

        # Update the small "Your answer" display
        self.answer_text.set_text(user_input)
        self.answer_text.set_color('black')

        # Update the "Feedback" label
        if is_correct:
            self.feedback_text.set_text('✓ CORRECT!')
            self.feedback_text.set_color('#27ae60')
        else:
            self.feedback_text.set_text(f'✗ WRONG (Correct: {correct_tone})')
            self.feedback_text.set_color('#e74c3c')
            
        # Update Main Button to "Next Item"
        self.btn_action.label.set_text('Next Item')
        self.btn_action.ax.set_facecolor('#27ae60') # Turns Green
        
        # Audio Playback
        self.status_text.set_text(f"Playing: {self.current_item['chinese']} ({self.current_item['pinyin']})")
        self.fig.canvas.draw_idle()
        
        threading.Thread(target=lambda: sd.play(self.current_audio_y, self.current_audio_sr), daemon=True).start()

    def _start_loop(self):
        """Loads the audio data and starts the cycling playback and SAI animation."""
        if not self.current_item: return

        # Combine correct path with filename
        audio_path = self.audio_base_path / self.current_item['audio']
        
        if audio_path.exists():
            try:
                # 1. Load and trim audio
                raw_audio, _ = librosa.load(str(audio_path), sr=self.sample_rate)
                self.audio_data, _ = librosa.effects.trim(raw_audio, top_db=25)
                
                # 2. Add silence padding for smoother looping
                self.audio_data = np.pad(self.audio_data, (0, 2000), 'constant') 
                
                # 3. Reset playback indices and start sound
                self.current_frame_index = 0
                self.is_playing = True
                sd.play(self.audio_data, self.sample_rate)
                
                # 4. Handle Timer for results
                if not self.timer_started:
                    self.question_start_time = time.time()
                    self.timer_started = True
                
                # 5. UI Updates: Transform button to "Check Answer"
                self.btn_action.label.set_text('Check Answer')
                self.btn_action.ax.set_facecolor('#3498db') # Turns Blue
                self.status_text.set_text('⟳ Looping Audio & SAI...')
                self.status_text.set_color('#3498db')
                self.fig.canvas.draw_idle()
                
            except Exception as e:
                print(f"Error starting loop: {e}")
                self.status_text.set_text('Audio Error!')
        else:
            print(f"Error: Audio file not found at {audio_path}")
            self.status_text.set_text('File Not Found!')

    def _handle_button_click(self, event):
        """Unified State Machine: Play -> Check -> Next"""
        if not self.is_playing and not self.answered:
            # Step 1: Start the audio and animation
            self._start_loop()
        elif not self.answered:
            # Step 2: Validate the user input
            # Call your check logic (make sure it reads self.text_input.text)
            self.check_answer(self.text_input.text) 
        else:
            # Step 3: Clear feedback and load new word
            self._next_word()

    def _check_answer_logic(self):
        """Validates input, provides feedback, and transforms button to 'Next'"""
        text = self.text_input.text.strip()
        if not text:
            self.status_text.set_text('⚠️ Please enter a tone number first!')
            self.fig.canvas.draw_idle()
            return

        # 1. Logic Processing
        user_answer = text.replace(' ', '').replace(',', '').replace('-', '')
        correct_answer = self.current_item['tone'].replace(',', '').replace('-', '')
        
        if self.question_start_time:
            self.question_elapsed_time = time.time() - self.question_start_time
        
        self.answered = True
        is_correct = (user_answer == correct_answer)
        
        # 2. Store Result
        self.results.append({
            'is_correct': is_correct,
            'user_answer': user_answer,
            'time_seconds': self.question_elapsed_time,
            'chinese': self.current_item['chinese'],
            'pinyin': self.current_item['pinyin'],
            'audio_file': self.current_item['audio'],
            'correct_tone': correct_answer
        })
        
        # 3. UI Feedback
        if is_correct:
            self.feedback_text.set_text('✓ CORRECT!')
            self.feedback_text.set_color('#27ae60')
        else:
            self.feedback_text.set_text(f'✗ INCORRECT (Correct: {correct_answer})')
            self.feedback_text.set_color('#e74c3c')
            
        self.answer_text.set_text(f"Target: {self.current_item['chinese']} ({self.current_item['pinyin']})")
        
        # 4. Button Transformation: Change to 'Next Word'
        self.btn_action.label.set_text('Next Word')
        self.btn_action.ax.set_facecolor('#27ae60') # Green
        self.status_text.set_text('Click Next Word to continue')
        self.status_text.set_color('#27ae60')
        
        self.fig.canvas.draw_idle()

    def _next_word(self):
        """Advances the quiz and resets the UI state"""
        self.question_count += 1
        if self.question_count >= self.max_questions:
            print("\n✓ Quiz completed!")
            self.is_playing = False
            sd.stop()
            self._save_results_to_file()
            plt.close(self.fig)
            self._launch_next_script()
        else:
            # Re-runs the selection logic which resets the button to 'Play Loop'
            self._select_random_item()

    def update_animation(self, frame):
        if not self.is_playing or self.audio_data is None:
            return [self.im_sai]

        # Handle Audio/Visual Sync Loop
        if self.current_frame_index + self.chunk_size < len(self.audio_data):
            # Advance to next chunk
            chunk = self.audio_data[self.current_frame_index : self.current_frame_index + self.chunk_size]
            self.current_frame_index += self.chunk_size
        else:
            # Loop Reset
            self.current_frame_index = 0
            chunk = self.audio_data[0 : self.chunk_size]
            
            # Restart Audio cleanly
            try:
                sd.stop()
                sd.play(self.audio_data, self.sample_rate)
            except: pass

        # Process Math
        nap_output = self.processor.process_chunk(chunk)
        sai_output = self.sai_processor.RunSegment(nap_output)
        self.vis.get_vowel_embedding(nap_output)
        self.vis.run_frame(sai_output)
        
        # Shift Image
        if self.vis.img.shape[1] > 1:
            self.vis.img[:, :-1] = self.vis.img[:, 1:]
            self.vis.draw_column(self.vis.img[:, -1])
        
        current_max = np.max(self.vis.img) if self.vis.img.size else 1
        self.im_sai.set_data(self.vis.img)
        self.im_sai.set_clim(vmin=0, vmax=max(1, min(255, current_max * 0.8)))
        
        return [self.im_sai]

    def toggle_play(self, event):
        if not self.current_item: return

        if self.is_playing:
            self.is_playing = False
            sd.stop()
            self.fig.canvas.draw_idle()
        else:
            self.is_playing = True
            
            # Combine correct path with filename
            audio_path = self.audio_base_path / self.current_item['audio']
            
            if audio_path.exists():
                raw_audio, _ = librosa.load(str(audio_path), sr=self.sample_rate)
                self.audio_data, _ = librosa.effects.trim(raw_audio, top_db=25)
                # Add silence padding for smoother looping
                self.audio_data = np.pad(self.audio_data, (0, 2000), 'constant') 
                
                self.current_frame_index = 0
                sd.play(self.audio_data, self.sample_rate)
                
                if not self.timer_started:
                    self.question_start_time = time.time()
                    self.timer_started = True
            else:
                print(f"Error: Audio file not found at {audio_path}")
            
            self.btn_play.label.set_text('■ Stop')
            self.btn_play.color = '#e74c3c'
            self.btn_play.hovercolor = '#c0392b'
            self.status_text.set_text('⟳ Looping Audio & SAI...')
            self.status_text.set_color('#3498db')
            self.fig.canvas.draw_idle()

    def _select_random_item(self):
        self.is_playing = False
        self.answered = False
        self.timer_started = False
        self.question_start_time = None
        sd.stop()

        # 1. CLEAR FEEDBACK LABELS (Delete previous results from screen)
        self.answer_text.set_text('')
        self.feedback_text.set_text('Feedback')
        self.feedback_text.set_color('#7f8c8d')  # Reset to neutral gray

        # 2. Reset Visuals and Inputs
        self.vis.img[:] = 0
        self.im_sai.set_data(self.vis.img)
        self.text_input.set_val('')
        
        # 3. Reset Button to Initial State
        self.btn_action.label.set_text('Play SAI')
        self.btn_action.ax.set_facecolor('#5B5FED') # Indigo
        
        self.status_text.set_text('Click Play Loop to start')
        self.status_text.set_color('#7f8c8d')
        
        # 4. Pick next item
        if not self.vocab_items: return
        random_item = random.choice(self.vocab_items)
        while random_item['id'] in self.used_words and len(self.used_words) < len(self.vocab_items):
            random_item = random.choice(self.vocab_items)
        
        self.current_item = random_item
        self.used_words.add(random_item['id'])
        
        self._update_progress()
        self.fig.canvas.draw_idle()

    def _update_progress(self):
        self.progress_text.set_text(f"Question {self.question_count + 1}/{self.max_questions}")

    def check_answer(self, text):
        if not self.current_item or self.answered: return
        
        user_answer = text.strip().replace(' ', '').replace(',', '').replace('-', '')
        correct_answer = self.current_item['tone'].replace(',', '').replace('-', '')
        
        if self.question_start_time:
            self.question_elapsed_time = time.time() - self.question_start_time
        
        self.answered = True
        is_correct = (user_answer == correct_answer)
        
        # --- MODIFIED BLOCK: Capture detailed info for the report ---
        result = {
            'is_correct': is_correct,
            'user_answer': user_answer,
            'time_seconds': self.question_elapsed_time,
            'chinese': self.current_item['chinese'],
            'pinyin': self.current_item['pinyin'],
            'audio_file': self.current_item['audio'],
            'correct_tone': correct_answer
        }
        self.results.append(result)
        # -----------------------------------------------------------
        
        if is_correct:
            self.feedback_text.set_text('CORRECT!')
            self.feedback_text.set_color('#27ae60')
        else:
            self.feedback_text.set_text(f'INCORRECT (Correct: {correct_answer})')
            self.feedback_text.set_color('#e74c3c')
            
        self.btn_action.label.set_text('Next Word')
        self.btn_action.ax.set_facecolor('#27ae60') # Green
        
        self.fig.canvas.draw_idle()

    def next_word(self, event):
        self.question_count += 1
        if self.question_count >= self.max_questions:
            print("\n✓ Quiz completed!")
            self.is_playing = False
            sd.stop()
            self._save_results_to_file()
            self.status_text.set_text('Quiz Completed!')
            self.status_text.set_color('blue')
            
            plt.close(self.fig)
            self._launch_next_script()
        else:
            self._select_random_item()

    def _save_results_to_file(self):
        try:
            script_dir = Path(__file__).parent
            results_dir = script_dir / 'result'
            results_dir.mkdir(exist_ok=True)
            timestamp_str = self.session_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')
            filename_ts = self.session_start_time.strftime('%Y%m%d_%H%M%S')
            filename = f"tone_quiz_sai_{filename_ts}.txt"
            filepath = results_dir / filename
            
            correct_count = sum(1 for r in self.results if r['is_correct'])
            total = len(self.results)
            
            # Build the report string
            lines = []
            lines.append("SPECTROGRAM TONE QUIZ RESULTS")
            lines.append("="*29)
            lines.append(f"Date: {timestamp_str}")
            lines.append(f"Score: {correct_count}/{total}")
            lines.append("")
            
            for i, res in enumerate(self.results):
                status = "CORRECT" if res['is_correct'] else "WRONG"
                lines.append(f"Q{i+1}: {res['chinese']} ({res['pinyin']})")
                lines.append(f"   Audio File: {res['audio_file']}")
                lines.append(f"   Correct Tone: {res['correct_tone']} | Your Answer: {res['user_answer']}")
                lines.append(f"   Result: {status}")
                lines.append(f"   Time: {res['time_seconds']:.2f}s")
                lines.append("-" * 30)
            
            report_content = "\n".join(lines)
            
            # Print to Console
            print("\n" + report_content)

            # Save to File
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            print(f"Saved result file to: {filepath}")
            
        except Exception as e:
            print(f"Save Error: {e}")

    def show(self):
        # IMPORTANT: Set blit=False to ensure buttons work reliably
        # Increased interval to 50ms (20fps) to reduce CPU load
        self.ani = animation.FuncAnimation(
            self.fig, self.update_animation, interval=50, blit=False, cache_frame_data=False
        )
        plt.show()

# -----------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------
if __name__ == '__main__':
    print("\n" + "="*60)
    print("MANDARIN TONE QUIZ (HQ LOOP VERSION)")
    print("="*60)
    
    intro = ToneIntroductionQuizWithSAI()
    intro.show()