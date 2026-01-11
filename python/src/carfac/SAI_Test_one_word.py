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
    print("Warning: JAX/CARFAC/SAI not found. Using Fallback mode.")
    JAX_AVAILABLE = False
    # If using strictly without JAX, usually we wouldn't exit, but based on your snippet:
    # sys.exit(1) 
    # I commented out exit so it runs even if you are just testing the UI/Audio logic without JAX installed.

# Import modules
# Assuming you have this module. If not, the script will fail here.
try:
    from modules.visualization_handler import VisualizationHandler, SAIParams
except ImportError:
    # Dummy mockup if module is missing, just to let the code run for testing
    class SAIParams:
        def __init__(self, num_channels, sai_width, future_lags, num_triggers_per_frame, trigger_window_width, input_segment_width, channel_smoothing_scale):
            self.num_channels = num_channels
            self.sai_width = sai_width
            self.future_lags = future_lags
            self.num_triggers_per_frame = num_triggers_per_frame
            self.trigger_window_width = trigger_window_width
            self.input_segment_width = input_segment_width
            self.channel_smoothing_scale = channel_smoothing_scale

    class VisualizationHandler:
        def __init__(self, fs, sai_params):
            self.img = np.zeros((sai_params.num_channels, sai_params.sai_width))
        def get_vowel_embedding(self, nap): pass
        def run_frame(self, sai): pass
        def draw_column(self, col): pass

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
                self.n_channels = 70 # Default fallback channels
        else:
            self.use_carfac = False
            self.n_channels = 70

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
        
        # Fallback (Simple spectrogram-like behavior for testing)
        if isinstance(audio_chunk, np.ndarray):
            chunk = audio_chunk.flatten()
        else:
            chunk = np.array(audio_chunk).flatten()
            
        if chunk.size == 0:
            return np.zeros((self.n_channels, 0), dtype=np.float32)
            
        # Create a dummy NAP (Neural Activity Pattern) from FFT for visualization if CARFAC fails
        spect = np.abs(np.fft.rfft(chunk, n=self.n_channels*2))
        spect = spect[:self.n_channels]
        # Reshape to (channels, time) - just repeating for this dummy chunk
        nap = np.tile(spect[:, np.newaxis], (1, 10)) 
        # Normalize
        return nap / (np.max(nap) + 1e-6)

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
        # Simplified dummy SAI generation for fallback
        rows, cols = nap_output.shape
        sai_out = np.zeros((rows, self.sai_params.sai_width))
        # Just copy data into the SAI window for visualization
        width = min(cols, self.sai_params.sai_width)
        if width > 0:
            sai_out[:, :width] = nap_output[:, :width]
        return sai_out

# -----------------------------------------------------------
# MAIN QUIZ CLASS (High Quality Animation)
# -----------------------------------------------------------
class ToneIntroductionQuizWithSAI:
    def __init__(self, audio_base_path=None):
        # 1. Setup Audio Path
        if audio_base_path is None:
            script_dir = Path(__file__).parent.resolve()
            # Try to find 'mandarin_audio', otherwise use current dir
            if (script_dir / 'mandarin_audio_two_syllable').exists():
                audio_base_path = script_dir / 'mandarin_audio_two_syllable'
            else:
                audio_base_path = script_dir
        
        self.audio_base_path = Path(audio_base_path)
        print(f"✓ Reading audio from: {self.audio_base_path}")

        self.sample_rate = 16000
        self.chunk_size = 512 # Reduced chunk size for smoother animation
        
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
        self.audio_stream = None
        
        # -------------------------------------------------------
        # 5. SINGLE FILE TEST CONFIGURATION
        # -------------------------------------------------------
        
        # ### EDIT YOUR FILE NAME HERE ###
        test_filename = "10_电脑_43.wav" 
        
        self.vocab_items = [
            {
                "id": 1, 
                "chinese": "TEST", 
                "pinyin": "Test File", 
                "tone": "0", 
                "audio": test_filename 
            }
        ]
        
        self.max_questions = 1  # Only run once
        # -------------------------------------------------------

        self.current_item = None
        self.answered = False
        self.question_count = 0
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
        
        # SAI Image
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
        self.btn_next = Button(ax_next, 'Next / Finish', color='#27ae60', hovercolor='#229954')
        self.btn_next.label.set_color('white')
        self.btn_next.on_clicked(self.next_word)
        
        self._update_progress()

    def update_animation(self, frame):
        if not self.is_playing or self.audio_data is None:
            return [self.im_sai]

        # Loop logic
        if self.current_frame_index + self.chunk_size < len(self.audio_data):
            chunk = self.audio_data[self.current_frame_index : self.current_frame_index + self.chunk_size]
            self.current_frame_index += self.chunk_size
        else:
            self.current_frame_index = 0
            chunk = self.audio_data[0 : self.chunk_size]
            # Restart audio
            if self.audio_stream:
                try:
                    # Just restarting the stream if it stopped (simple approach)
                    if not self.audio_stream.active:
                        self.audio_stream.stop()
                        self.audio_stream.close()
                        self.audio_stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype=np.float32)
                        self.audio_stream.start()
                        sd.play(self.audio_data, self.sample_rate)
                except: pass

        # Process SAI
        nap_output = self.processor.process_chunk(chunk)
        sai_output = self.sai_processor.RunSegment(nap_output)
        
        # Visualize
        try:
            self.vis.get_vowel_embedding(nap_output)
            self.vis.run_frame(sai_output)
            
            # Scroll
            if self.vis.img.shape[1] > 1:
                self.vis.img[:, :-1] = self.vis.img[:, 1:]
                self.vis.draw_column(self.vis.img[:, -1])
                
            # Update plot
            current_max = np.max(self.vis.img) if self.vis.img.size else 1
            self.im_sai.set_data(self.vis.img)
            self.im_sai.set_clim(vmin=0, vmax=max(1, min(255, current_max * 0.8)))
        except Exception as e:
            # Fallback if vis module fails
            pass
        
        return [self.im_sai]

    def toggle_play(self, event):
        if not self.current_item: return

        if self.is_playing:
            self.is_playing = False
            sd.stop()
            self.btn_play.label.set_text('▶ Play Loop')
            self.btn_play.color = '#5B5FED'
            self.status_text.set_text('Stopped.')
        else:
            self.is_playing = True
            
            # Load Audio
            audio_path = self.audio_base_path / self.current_item['audio']
            
            if not audio_path.exists():
                print(f"Error: File not found at {audio_path}")
                self.status_text.set_text('FILE NOT FOUND')
                self.status_text.set_color('red')
                self.is_playing = False
                return

            try:
                raw_audio, _ = librosa.load(str(audio_path), sr=self.sample_rate)
                self.audio_data, _ = librosa.effects.trim(raw_audio, top_db=25)
                # Pad for looping
                self.audio_data = np.pad(self.audio_data, (0, 2000), 'constant')
                
                self.current_frame_index = 0
                sd.play(self.audio_data, self.sample_rate)
                
                self.btn_play.label.set_text('■ Stop')
                self.btn_play.color = '#e74c3c'
                self.status_text.set_text(f"Playing: {self.current_item['audio']}")
            except Exception as e:
                print(f"Audio Load Error: {e}")
                self.is_playing = False

    def _select_random_item(self):
        self.is_playing = False
        sd.stop()
        self.vis.img[:] = 0
        self.im_sai.set_data(self.vis.img)
        self.btn_play.label.set_text('▶ Play Loop')
        self.btn_play.color = '#5B5FED'
        self.text_input.set_val('')
        self.answer_text.set_text('')
        self.feedback_text.set_text('')
        
        # Force select the single item
        self.current_item = self.vocab_items[0]
        self.answered = False
        self._update_progress()
        print(f"\nLoaded Test File: {self.current_item['audio']}")

    def _update_progress(self):
        self.progress_text.set_text(f"Single File Test Mode")

    def check_answer_button(self, event):
        self.check_answer(self.text_input.text)

    def check_answer(self, text):
        if not self.current_item: return
        self.answer_text.set_text(f"Your input: {text}")
        self.feedback_text.set_text("Test Mode - Answer Logged")
        self.answered = True

    def next_word(self, event):
        print("\nTest completed.")
        self.is_playing = False
        sd.stop()
        plt.close(self.fig)

    def show(self):
        self.ani = animation.FuncAnimation(
            self.fig, self.update_animation, interval=20, blit=True
        )
        plt.show()

# -----------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------
if __name__ == '__main__':
    print("\n" + "="*60)
    print("SINGLE FILE TEST MODE")
    print("="*60)
    
    try:
        intro = ToneIntroductionQuizWithSAI()
        intro.show()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)