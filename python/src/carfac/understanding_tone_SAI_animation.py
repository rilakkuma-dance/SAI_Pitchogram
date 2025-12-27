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
            possible_paths = [script_dir / 'carfac', script_dir.parent / 'carfac', script_dir / 'carfac']
            
            audio_base_path = None
            for path in possible_paths:
                if path.exists():
                    audio_base_path = path
                    print(f"✓ Found audio path: {audio_base_path}")
                    break
            if audio_base_path is None:
                audio_base_path = script_dir / 'mandarin_audio'
        
        self.audio_base_path = Path(audio_base_path)
        self.sample_rate = 16000
        self.chunk_size = 512
        
        # 2. Control Flags
        self.is_playing = False  # Controls audio playback logic
        
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
        self.audio_stream = None
        
        # 5. Quiz Data
        self.vocab_items = [
            {"id": 1, "chinese": "书", "pinyin": "shū", "tone": "1", "audio": "mandarin_audio/01_书_1.mp3"},
            {"id": 2, "chinese": "女人", "pinyin": "nǚrén", "tone": "32", "audio": "mandarin_audio/02_女人_32.mp3"},
            {"id": 3, "chinese": "雄", "pinyin": "xióng", "tone": "2", "audio": "mandarin_audio/03_雄_2.mp3"},
            {"id": 4, "chinese": "去", "pinyin": "qù", "tone": "4", "audio": "mandarin_audio/04_去_4.mp3"},
            {"id": 6, "chinese": "喜欢", "pinyin": "xǐhuān", "tone": "31", "audio": "mandarin_audio/06_喜欢_31.mp3"},
            {"id": 7, "chinese": "街道", "pinyin": "jiēdào", "tone": "14", "audio": "mandarin_audio/07_街道_14.mp3"},
            {"id": 8, "chinese": "熊猫", "pinyin": "xióngmāo", "tone": "21", "audio": "mandarin_audio/08_熊猫_21.mp3"},
            {"id": 9, "chinese": "书店", "pinyin": "shūdiàn", "tone": "14", "audio": "mandarin_audio/09_书店_14.mp3"},
            {"id": 10, "chinese": "去年", "pinyin": "qùnián", "tone": "42", "audio": "mandarin_audio/10_去年_42.mp3"},
            {"id": 11, "chinese": "中午", "pinyin": "zhōngwǔ", "tone": "13", "audio": "mandarin_audio/11_中午_13.mp3"},
            {"id": 12, "chinese": "老师", "pinyin": "lǎoshī", "tone": "31", "audio": "mandarin_audio/12_老师_31.mp3"},
            {"id": 13, "chinese": "学校", "pinyin": "xuéxiào", "tone": "24", "audio": "mandarin_audio/13_学校_24.mp3"},
            {"id": 14, "chinese": "医院", "pinyin": "yīyuàn", "tone": "14", "audio": "mandarin_audio/14_医院_14.mp3"},
            {"id": 15, "chinese": "游戏", "pinyin": "yóuxì", "tone": "24", "audio": "mandarin_audio/15_游戏_24.mp3"},
            {"id": 16, "chinese": "她", "pinyin": "tā", "tone": "1", "audio": "mandarin_audio/16_她_1.mp3"},
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
        self._select_random_item()
        
    def _setup_interface(self):
        main_ax = self.fig.add_axes([0.1, 0.05, 0.8, 0.9])
        main_ax.axis('off')
        
        # SAI Image (High Quality Settings)
        self.ax_sai = self.fig.add_axes([0.15, 0.42, 0.7, 0.38])
        self.im_sai = self.ax_sai.imshow(
            self.vis.img, aspect='auto', origin='upper', cmap='inferno', 
            interpolation='bilinear', extent=[0, self.sai_params.sai_width, 0, self.n_channels],
            vmin=0, vmax=255
        )
        self.ax_sai.axis('off')
        
        # Labels
        self.progress_text = main_ax.text(0.5, 0.82, '', fontsize=11, ha='center', weight='bold', color='#7f8c8d')
        self.status_text = main_ax.text(0.5, 0.30, 'Click Play to start loop', fontsize=9, ha='center', color='#7f8c8d')
        
        # Play Button
        ax_play = plt.axes([0.35, 0.34, 0.3, 0.05])
        self.btn_play = Button(ax_play, '▶ Play Loop', color='#5B5FED', hovercolor='#4B4FDD')
        self.btn_play.label.set_color('white')
        self.btn_play.label.set_weight('bold')
        self.btn_play.on_clicked(self.toggle_play)
        
        # Inputs
        main_ax.text(0.5, 0.25, 'Enter Tone Numbers', fontsize=10, ha='center', color='#666666', weight='bold')
        main_ax.text(0.5, 0.21, '(Example: for "tiānqì" type "14")', fontsize=9, ha='center', color='#999999')
        ax_input = plt.axes([0.3, 0.16, 0.4, 0.05])
        self.text_input = TextBox(ax_input, '', initial='', color='white', hovercolor='#f9f9f9')
        
        self.answer_text = main_ax.text(0.5, 0.12, '', fontsize=12, ha='center', weight='bold', color='#34495e')
        self.feedback_text = main_ax.text(0.5, 0.08, '', fontsize=14, ha='center', weight='bold')
        
        # Control Buttons
        ax_check = plt.axes([0.15, 0.01, 0.3, 0.04])
        self.btn_check = Button(ax_check, 'Check Answer', color='#3498db', hovercolor='#2980b9')
        self.btn_check.label.set_color('white')
        self.btn_check.on_clicked(self.check_answer_button)
        
        ax_next = plt.axes([0.55, 0.01, 0.3, 0.04])
        self.btn_next = Button(ax_next, 'Next Word', color='#27ae60', hovercolor='#229954')
        self.btn_next.label.set_color('white')
        self.btn_next.on_clicked(self.next_word)
        
        self._update_progress()

    def update_animation(self, frame):
        """This function is called by FuncAnimation 60 times per second"""
        if not self.is_playing or self.audio_data is None:
            return [self.im_sai] # Do nothing if paused

        # Calculate how many chunks to process to keep up with real-time
        # (Simplified: we process one chunk per animation frame)
        if self.current_frame_index + self.chunk_size < len(self.audio_data):
            chunk = self.audio_data[self.current_frame_index : self.current_frame_index + self.chunk_size]
            self.current_frame_index += self.chunk_size
        else:
            # Loop back to start
            self.current_frame_index = 0
            chunk = self.audio_data[0 : self.chunk_size]
            
            # Restart audio playback for syncing sound
            if self.audio_stream:
                try:
                    self.audio_stream.stop()
                    self.audio_stream.close()
                except: pass
            
            try:
                self.audio_stream = sd.OutputStream(
                    samplerate=self.sample_rate, channels=1, dtype=np.float32
                )
                self.audio_stream.start()
                # We simply play the whole buffer again in background
                sd.play(self.audio_data, self.sample_rate)
            except: pass

        # Process SAI (High Quality)
        nap_output = self.processor.process_chunk(chunk)
        sai_output = self.sai_processor.RunSegment(nap_output)
        self.vis.get_vowel_embedding(nap_output)
        self.vis.run_frame(sai_output)
        
        # Scroll Buffer
        if self.vis.img.shape[1] > 1:
            self.vis.img[:, :-1] = self.vis.img[:, 1:]
            self.vis.draw_column(self.vis.img[:, -1])
        
        # Update Image
        current_max = np.max(self.vis.img) if self.vis.img.size else 1
        self.im_sai.set_data(self.vis.img)
        self.im_sai.set_clim(vmin=0, vmax=max(1, min(255, current_max * 0.8)))
        
        return [self.im_sai]

    def toggle_play(self, event):
        """Toggle between Playing Loop and Stopping"""
        if not self.current_item: return

        if self.is_playing:
            # STOP
            self.is_playing = False
            sd.stop() # Stop sound
            self.btn_play.label.set_text('▶ Play Loop')
            self.btn_play.color = '#5B5FED'
            self.btn_play.hovercolor = '#4B4FDD'
            self.status_text.set_text('Stopped.')
        else:
            # START
            self.is_playing = True
            
            # Load Audio
            audio_path = self.audio_base_path / self.current_item['audio']
            if audio_path.exists():
                raw_audio, _ = librosa.load(str(audio_path), sr=self.sample_rate)
                # Trim silence
                self.audio_data, _ = librosa.effects.trim(raw_audio, top_db=25)
                # Pad slightly to ensure smooth loop
                self.audio_data = np.pad(self.audio_data, (0, 2000), 'constant') 
                
                self.current_frame_index = 0
                
                # Start initial sound
                sd.play(self.audio_data, self.sample_rate)
                
                # Start Timer
                if not self.timer_started:
                    self.question_start_time = time.time()
                    self.timer_started = True
            
            self.btn_play.label.set_text('■ Stop')
            self.btn_play.color = '#e74c3c'
            self.btn_play.hovercolor = '#c0392b'
            self.status_text.set_text('⟳ Looping Audio & SAI...')
            self.status_text.set_color('#3498db')

    def _select_random_item(self):
        """Select a random vocabulary item"""
        # Reset state
        self.is_playing = False
        sd.stop()
        self.vis.img[:] = 0
        self.im_sai.set_data(self.vis.img)
        self.btn_play.label.set_text('▶ Play Loop')
        self.btn_play.color = '#5B5FED'
        self.text_input.set_val('')
        self.answer_text.set_text('')
        self.feedback_text.set_text('')
        self.status_text.set_text('Click Play Loop to start')
        self.status_text.set_color('#7f8c8d')
        
        # Pick new word
        random_item = random.choice(self.vocab_items)
        while random_item['id'] in self.used_words and len(self.used_words) < len(self.vocab_items):
            random_item = random.choice(self.vocab_items)
        
        self.current_item = random_item
        self.used_words.add(random_item['id'])
        self.answered = False
        self.timer_started = False
        self.question_start_time = None
        
        self._update_progress()
        print(f"\nTarget: {self.current_item['pinyin']} (Tone: {self.current_item['tone']})")

    def _update_progress(self):
        self.progress_text.set_text(f"Question {self.question_count + 1}/{self.max_questions}")

    def check_answer_button(self, event):
        text = self.text_input.text
        if not text.strip(): return
        self.check_answer(text)

    def check_answer(self, text):
        if not self.current_item or self.answered: return
        
        user_answer = text.strip().replace(' ', '').replace(',', '').replace('-', '')
        correct_answer = self.current_item['tone'].replace(',', '').replace('-', '')
        
        if self.question_start_time:
            self.question_elapsed_time = time.time() - self.question_start_time
        
        self.answered = True
        is_correct = (user_answer == correct_answer)
        
        result = {
            'is_correct': is_correct,
            'user_answer': user_answer,
            'time_seconds': round(self.question_elapsed_time, 2)
        }
        self.results.append(result)
        
        self.answer_text.set_text(f"Your answer: {user_answer}")
        if is_correct:
            self.feedback_text.set_text('✓ CORRECT!')
            self.feedback_text.set_color('#27ae60')
        else:
            self.feedback_text.set_text(f'✗ INCORRECT (Correct: {correct_answer})')
            self.feedback_text.set_color('#e74c3c')

    def next_word(self, event):
        self.question_count += 1
        if self.question_count >= self.max_questions:
            print("\n✓ Quiz completed!")
            self.is_playing = False
            sd.stop()
            self._save_results_to_file()
            self.status_text.set_text('Quiz Completed!')
            self.status_text.set_color('blue')
        else:
            self._select_random_item()

    def _save_results_to_file(self):
        try:
            script_dir = Path(__file__).parent
            results_dir = script_dir / 'result'
            results_dir.mkdir(exist_ok=True)
            timestamp = self.session_start_time.strftime('%Y%m%d_%H%M%S')
            filename = f"tone_quiz_sai_{timestamp}.txt"
            filepath = results_dir / filename
            
            correct_count = sum(1 for r in self.results if r['is_correct'])
            total = len(self.results)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"MANDARIN QUIZ RESULTS\nDate: {timestamp}\n")
                f.write(f"Score: {correct_count}/{total}\n\n")
                for i, res in enumerate(self.results):
                    f.write(f"Q{i+1}: {'Correct' if res['is_correct'] else 'Incorrect'} (Ans: {res['user_answer']})\n")
            print(f"Saved: {filepath}")
        except Exception as e:
            print(f"Save Error: {e}")

    def show(self):
        # This is the High Quality Engine (FuncAnimation)
        self.ani = animation.FuncAnimation(
            self.fig, self.update_animation, interval=20, blit=True
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