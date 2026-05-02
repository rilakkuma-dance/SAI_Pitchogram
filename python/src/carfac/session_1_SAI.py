import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import sys
import numpy as np
from pathlib import Path
import sounddevice as sd
import librosa
import random
import os
import time
import subprocess 
import csv 
from datetime import datetime

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

# ==========================================
# IMPORT THE NEW CONFIGURATION HELPER
# ==========================================
from sai_config import get_sai_params
# ==========================================

# ==========================================
# FIX: CONFIGURE FONTS FOR CHINESE SUPPORT
# ==========================================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

# JAX/CARFAC/SAI imports
try:
    sys.path.append('./jax')
    import jax
    import jax.numpy as jnp
    import carfac.jax.carfac as carfac
    import sai
    JAX_AVAILABLE = True
except ImportError:
    print("Warning: JAX/CARFAC/SAI not found. Install required packages.")
    JAX_AVAILABLE = False
    sys.exit(1)

from modules.visualization_handler import VisualizationHandler

# -----------------------------------------------------------
# Audio Processor Class
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
        print(f"Total Channels: {self.n_channels}")

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
# MAIN QUIZ CLASS
# -----------------------------------------------------------
class ToneIntroductionQuizMixed:
    # --- Fix 5: Restore TONE_SHAPES so answer logic doesn't crash ---
    TONE_SHAPES = {1: '―', 2: '╱', 3: '∨', 4: '╲'}
    
    # --- 2-Color Duo-Tone Scheme ---
    TONE_COLORS = {
        1: '#005B96',  # Deep Accessible Blue
        2: '#005B96',  # Charcoal Gray
        3: '#005B96',  # Deep Accessible Blue
        4: '#005B96',  # Charcoal Gray
    }

    # --- Fix 4: Make paths absolute relative to the script's directory ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_DIR = os.path.join(SCRIPT_DIR, 'pitchogram_screenshot')

    TONE_IMAGES = {
        1: os.path.join(IMAGE_DIR, 'tone1.png'),
        2: os.path.join(IMAGE_DIR, 'tone2.png'),
        3: os.path.join(IMAGE_DIR, 'tone3.png'),
        4: os.path.join(IMAGE_DIR, 'tone4.png')
    }

    # Single folder containing all .wav files (1- and 2-syllable mixed)
    AUDIO_FOLDER = 'mandarin_audio'

    def __init__(self):
        self.script_dir = Path(__file__).parent.resolve()
        self.sample_rate = 16000
        self.chunk_size = 450
        
        # 1. Setup single folder
        self.audio_folder = self._find_folder(self.AUDIO_FOLDER)
        if not self.audio_folder:
            print(f"ERROR: Folder '{self.AUDIO_FOLDER}' not found "
                  f"in {self.script_dir} or its parent.")
        
        # 2. Scan the folder; syllable count is inferred from each filename
        all_items = self._scan_folder(self.audio_folder)
        items_one = [it for it in all_items if it['syllables'] == 1]
        items_two = [it for it in all_items if it['syllables'] == 2]

        random.shuffle(items_one)
        random.shuffle(items_two)

        # Pick 3 of each (or as many as available)
        selected_one = items_one[:15]
        selected_two = items_two[:15]
        
        self.vocab_items = selected_one + selected_two
        random.shuffle(self.vocab_items)
            
        print(f"Loaded {len(self.vocab_items)} files "
              f"({len(selected_one)} 1-syllable, {len(selected_two)} 2-syllable) "
              f"from '{self.AUDIO_FOLDER}'.")
        
        # 3. Control Flags
        self.is_playing = False
        
        # 4. SAI Setup
        self.processor = AudioProcessor(fs=self.sample_rate)
        self.n_channels = self.processor.n_channels
        
        self.sai_params = get_sai_params(self.n_channels, self.chunk_size)
        self.sai_processor = SAIProcessor(self.sai_params)
        self.vis = VisualizationHandler(self.sample_rate, self.sai_params)
        
        # RGB Buffer
        self.rgb_img = np.zeros((self.n_channels, 400, 3), dtype=np.float32)

        # 5. Audio Playback Variables
        self.audio_data = None
        self.current_frame_index = 0
        
        # 6. Quiz State
        self.current_item = None
        self.answered = False
        self.question_count = 0
        self.max_questions = len(self.vocab_items)
        self.results = [] 

        # Tone selection state — list of integers in selection order
        self.selected_tones = []
        
        self.fig = plt.figure(figsize=(11, 10))
        self.fig.patch.set_facecolor('white')
        self.fig.canvas.manager.set_window_title("")       
        
        self.timer_started = False
        self.question_start_time = None
        
        self._setup_interface()

        # Keyboard: digits select tones, 'p' toggles to production, etc.
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        if self.vocab_items:
            self._select_next_item()
        else:
            print("ERROR: No audio files found.")

    def _find_folder(self, folder_name):
        path = self.script_dir / folder_name
        if path.exists(): return path
        path = self.script_dir.parent / folder_name
        if path.exists(): return path
        return None

    def _scan_folder(self, folder_path):
        """Scan one folder containing .wav files of the form
        '<idx>_<chinese>_<tone>.wav', e.g. '01_天_1.wav' or '01_中国_12.wav'.
        The number of digits in <tone> determines the syllable count.
        """
        items = []
        if not folder_path:
            return items
        
        for file_path in folder_path.glob('*.wav'):
            try:
                parts = file_path.stem.split('_')
                if len(parts) < 3:
                    print(f"Skipping (unexpected name): {file_path.name}")
                    continue

                tone = parts[-1]
                chinese = parts[-2]

                # Syllable count = number of digit characters in the tone field.
                # Examples: '1' -> 1, '12' -> 2, '4-3' -> 2.
                tone_digits = ''.join(ch for ch in tone if ch.isdigit())
                if not tone_digits:
                    print(f"Skipping (no tone digits): {file_path.name}")
                    continue
                syllables = len(tone_digits)

                items.append({
                    "id": file_path.name,
                    "chinese": chinese,
                    "tone": tone_digits,         # normalized: digits only
                    "audio_path": file_path,
                    "syllables": syllables,
                })
            except Exception as e:
                print(f"Skipping {file_path.name}: {e}")
        return items

    # ---------------------------------------------------------------
    # UI: matches the sketch — pitchogram on top, "Identify the tone(s)",
    # four tone buttons [1 ―] [2 ╱] [3 ∨] [4 ╲], and an "Answer" box.
    # ---------------------------------------------------------------
    def _setup_interface(self):
        self.ax_ui = self.fig.add_axes([0, 0, 1, 1])
        self.ax_ui.axis('off')

        # --- Pitchogram (the SAI display, framed like in the sketch) ---
        self.ax_sai = self.fig.add_axes([0.10, 0.55, 0.80, 0.36])
        self.im_sai = self.ax_sai.imshow(
            self.rgb_img,
            aspect='auto', origin='upper',
            extent=[0, 11.25, self.processor.n_channels, 0]
        )
        self.ax_sai.set_xticks([])
        self.ax_sai.set_yticks([])
        for spine in self.ax_sai.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#222')
            spine.set_linewidth(2)

        # "pitchogram" title
        self.ax_ui.text(
            0.5, 0.935, 'pitchogram',
            ha='center', va='center', fontsize=30, weight='bold', color='#222'
        )

        # --- Mode indicator (small, top-right) ---

        # --- Prompt: "Identify the tone(s)" ---
        self.prompt_text = self.ax_ui.text(
            0.5, 0.48, 'Identify the tone(s)',
            ha='center', va='center',
            fontsize=30, weight='bold', color='#222'
        )

        # --- Status / progress (small, just under the prompt) ---
        self.status_text = self.ax_ui.text(
            0.5, 0.435, 'Click Play to listen',
            ha='center', va='center',
            fontsize=20, color='#7f8c8d'
        )
        self.progress_text = self.ax_ui.text(
            0.5, 0.405, '', ha='center', va='center',
            fontsize=20, color='#7f8c8d'
        )

        # --- Four tone buttons in a row ---
        n_buttons = 4
        total_width = 0.16       # Total width for the button + image combo
        button_height = 0.10
        side_margin = 0.06
        gap = (1.0 - 2 * side_margin - n_buttons * total_width) / (n_buttons - 1)
        button_y = 0.23

        self.tone_buttons = {}
        self.tone_button_axes = {}
        self.tone_image_axes = {} # Store image axes to prevent garbage collection

        for i, tone_num in enumerate([1, 2, 3, 4]):
            x = side_margin + i * (total_width + gap)
            color = self.TONE_COLORS[tone_num]
            
            # Divide the total space: 40% for the button (number), 60% for the image
            btn_w = total_width * 0.40
            img_w = total_width * 0.60
            
            # --- The Button Axes ---
            ax_btn = self.fig.add_axes([x, button_y, btn_w, button_height])
            label = f"{tone_num}"
            btn = Button(ax_btn, label, color=color, hovercolor=self._lighten(color))
            btn.label.set_fontsize(30)
            btn.label.set_weight('bold')
            btn.label.set_color('white')
            btn.on_clicked(lambda event, t=tone_num: self._on_tone_button(t))
            
            self.tone_buttons[tone_num] = btn
            self.tone_button_axes[tone_num] = ax_btn

            # --- The Image Axes (Positioned strictly next to it) ---
            img_x = x + btn_w
            ax_img = self.fig.add_axes([img_x, button_y, img_w, button_height])
            ax_img.axis('off')
            
            # Give the image axes the same background color
            ax_img.patch.set_visible(True)
            ax_img.patch.set_facecolor(color)

            # Load and draw the image
            try:
                img_path = self.TONE_IMAGES[tone_num]
                if os.path.exists(img_path):
                    img = plt.imread(img_path)
                    ax_img.imshow(img, aspect='equal')
                else:
                    print(f"Warning: Image file not found at {img_path}")
            except Exception as e:
                print(f"Error loading image for tone {tone_num}: {e}")

            # --- Make the image clickable too ---
            def make_img_clickable(event, t=tone_num, axis=ax_img):
                if event.inaxes == axis:
                    self._on_tone_button(t)
                    
            self.fig.canvas.mpl_connect('button_press_event', make_img_clickable)
            self.tone_image_axes[tone_num] = ax_img

        # --- Answer box ---
        self.answer_text = self.ax_ui.text(
            0.5, 0.13, 'Answer:  [ _ ]',
            ha='center', va='center',
            fontsize=30, weight='bold', color='#222',
            bbox=dict(
                boxstyle='round,pad=0.6',
                facecolor='#f4f4f4',
                edgecolor='#888',
                linewidth=2
            )
        )

        # --- Feedback text ---
        self.feedback_text = self.ax_ui.text(
            0.5, 0.075, '',
            ha='center', va='center',
            fontsize=20, weight='bold', color='#7f8c8d'
        )

        # --- Bottom row: Play / Next / Mode buttons ---
        self.ax_play_btn = plt.axes([0.18, 0.005, 0.22, 0.045])
        self.btn_action = Button(self.ax_play_btn, '▶ Play', color='#3498db', hovercolor='#5dade2')
        self.btn_action.label.set_fontfamily('Segoe UI Symbol') 
        self.btn_action.label.set_color('white')
        self.btn_action.label.set_weight('bold')
        self.btn_action.label.set_fontsize(20)
        self.btn_action.on_clicked(self._handle_play_click)

        self.ax_next_btn = plt.axes([0.42, 0.005, 0.18, 0.045])
        self.btn_next = Button(self.ax_next_btn, 'Next →', color='#bdc3c7', hovercolor='#95a5a6')
        self.btn_next.label.set_color('#222')
        self.btn_next.label.set_weight('bold')
        self.btn_next.label.set_fontsize(20)
        self.btn_next.on_clicked(lambda event: self._next_word())

        self.ax_mode_btn = plt.axes([0.62, 0.005, 0.22, 0.045])
        self.btn_mode = Button(self.ax_mode_btn, 'Production Mode', color='#444466', hovercolor='#6666aa')
        self.btn_mode.label.set_color('white')
        self.btn_mode.label.set_weight('bold')
        self.btn_mode.label.set_fontsize(20)
        self.btn_mode.on_clicked(lambda event: self._switch_to_production())

    @staticmethod
    def _lighten(hex_color, amount=0.18):
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        new_rgb = tuple(min(255, int(c + (255 - c) * amount)) for c in rgb)
        return '#{:02x}{:02x}{:02x}'.format(*new_rgb)

    # ---------------------------------------------------------------
    # Answer / Tone selection logic
    # ---------------------------------------------------------------
    def _update_answer_display(self):
        if not self.current_item:
            self.answer_text.set_text('Answer:  [ _ ]')
            return

        n_syllables = self.current_item.get('syllables', 1)

        if not self.selected_tones:
            if n_syllables == 1:
                self.answer_text.set_text('Answer:  [ _ ]')
            else:
                self.answer_text.set_text('Answer:  [ _ ] - [ _ ]')
            return

        parts = []
        for t in self.selected_tones:
            parts.append(f"[{t} {self.TONE_SHAPES.get(t, '?')}]")
        while len(parts) < n_syllables:
            parts.append('[ _ ]')
        self.answer_text.set_text('Answer:  ' + ' - '.join(parts))

    def _on_tone_button(self, tone_number):
        if not self.current_item:
            return
        if self.answered:
            return

        n_syllables = self.current_item.get('syllables', 1)

        if len(self.selected_tones) >= n_syllables:
            self.selected_tones = [tone_number]
        else:
            self.selected_tones.append(tone_number)

        self._update_answer_display()

        if len(self.selected_tones) == n_syllables:
            self._check_answer()

        self.fig.canvas.draw_idle()

    def _check_answer(self):
        if not self.current_item or self.answered:
            return

        user_answer_str = ''.join(str(t) for t in self.selected_tones)
        correct_answer = self.current_item['tone'].replace(',', '').replace('-', '')

        time_taken = 0.0
        if self.question_start_time:
            time_taken = time.time() - self.question_start_time

        self.answered = True
        is_correct = (user_answer_str == correct_answer)

        self.results.append({
            'question_idx': self.question_count + 1,
            'chinese': self.current_item['chinese'],
            'syllables': self.current_item['syllables'],
            'correct_tone': correct_answer,
            'user_answer': user_answer_str,
            'is_correct': is_correct,
            'time_seconds': round(time_taken, 2),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        self.status_text.set_text(f'Revealed: {self.current_item["chinese"]} (Tone {correct_answer})')
        self.status_text.set_color('#555555')

        if is_correct:
            self.feedback_text.set_text('✓ CORRECT')
            self.feedback_text.set_fontfamily('Segoe UI Symbol') # Corrected!
            self.feedback_text.set_color('#27ae60')
        else:
            correct_tones = [int(c) for c in correct_answer if c.isdigit()]
            correct_display = ' - '.join(
                f"[{t} {self.TONE_SHAPES.get(t, '?')}]" for t in correct_tones
            )
            self.feedback_text.set_text(f'✗ INCORRECT — Correct: {correct_display}')
            self.feedback_text.set_fontfamily('Segoe UI Symbol') # Corrected!
            self.feedback_text.set_color('#e74c3c')

        self.btn_action.label.set_text('Next Question')
        self.btn_action.ax.set_facecolor('#27ae60')

    def _handle_play_click(self, event):
        if self.answered:
            self._next_word()
            return
        self._start_loop()

    def _start_loop(self):
        if not self.current_item:
            return
        audio_path = self.current_item['audio_path']
        if audio_path.exists():
            try:
                raw_audio, _ = librosa.load(str(audio_path), sr=self.sample_rate)
                self.audio_data, _ = librosa.effects.trim(raw_audio, top_db=25)
                self.audio_data = np.pad(self.audio_data, (0, 3000), 'constant')
                self.current_frame_index = 0
                self.is_playing = True
                sd.play(self.audio_data, self.sample_rate)

                if not self.timer_started:
                    self.question_start_time = time.time()
                    self.timer_started = True

                self.btn_action.label.set_text('▶ Replay')
                self.btn_action.ax.set_facecolor('#3498db')

                self.status_text.set_text('Playing… choose a tone')
                self.status_text.set_color('#3498db')
            except Exception as e:
                print(f"Playback error: {e}")

    def _next_word(self):
        self.question_count += 1
        if self.question_count >= self.max_questions or self.question_count >= len(self.vocab_items):
            print("Quiz Completed.")
            self.is_playing = False
            sd.stop()
            self._save_results_to_file()
            plt.close(self.fig)
        else:
            self._select_next_item()

    def _save_results_to_file(self):
        filename = "session1_SAI_results.csv"
        filepath = self.script_dir / filename
        file_exists = filepath.exists()

        try:
            with open(filepath, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=[
                    'question_idx', 'chinese', 'syllables', 'correct_tone',
                    'user_answer', 'is_correct', 'time_seconds', 'timestamp'
                ])
                if not file_exists:
                    writer.writeheader()
                writer.writerows(self.results)
            print(f"Results appended to {filepath}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    def _select_next_item(self):
        self.is_playing = False
        self.answered = False
        self.timer_started = False
        self.question_start_time = None
        self.selected_tones = []

        sd.stop()
        self.rgb_img[:] = 0
        self.vis.img[:] = 0
        self.im_sai.set_data(self.rgb_img)

        self.feedback_text.set_text('')
        self.current_item = self.vocab_items[self.question_count]

        self.btn_action.label.set_text('▶ Play')
        self.btn_action.ax.set_facecolor('#3498db')

        self.status_text.set_text('Click Play to listen')
        self.status_text.set_color('#7f8c8d')

        self._update_answer_display()
        self._update_progress()

    def _update_progress(self):
        n_syl = self.current_item.get('syllables', 1) if self.current_item else 1
        progress = f"Question {self.question_count + 1} / {self.max_questions}"
        if n_syl > 1:
            progress += "    (2 syllables — pick tones in order)"
        self.progress_text.set_text(progress)

    # ---------------------------------------------------------------
    # Keyboard shortcuts
    # ---------------------------------------------------------------
    def _on_key_press(self, event):
        if event.key in ('1', '2', '3', '4'):
            self._on_tone_button(int(event.key))
        elif event.key == ' ':
            if self.answered:
                self._next_word()
            else:
                self._start_loop()
        elif event.key in ('n', 'right', 'enter'):
            if self.answered:
                self._next_word()
        elif event.key in ('p', 'P'):
            self._switch_to_production()
        elif event.key == 'backspace':
            if self.selected_tones and not self.answered:
                self.selected_tones.pop()
                self._update_answer_display()
                self.fig.canvas.draw_idle()

    def _switch_to_production(self):
        """Close this app and launch the production mode script."""
        print("Switching to Production Mode…")
        self.status_text.set_text('Switching to Production mode…')
        self.status_text.set_color('#e67e22')
        self.fig.canvas.draw_idle()
        plt.pause(0.4)

        sd.stop()
        if self.results:
            self._save_results_to_file()

        candidates = [
            r"C:\Users\z5718263\SAI_Pitchogram\python\src\carfac\session_2_SAI.py"
        ]
        next_script = None
        for name in candidates:
            cand = self.script_dir / name
            if cand.exists():
                next_script = cand
                break
            cand = self.script_dir.parent / name
            if cand.exists():
                next_script = cand
                break

        if next_script:
            print(f"🚀 Launching: {next_script}")
            subprocess.Popen([sys.executable, str(next_script)])
        else:
            print(f"⚠️ Could not find a production script. Tried: {candidates}")

        plt.close(self.fig)

    # ---------------------------------------------------------------
    # Animation
    # ---------------------------------------------------------------
    def update_animation(self, frame):
        if not self.is_playing or self.audio_data is None:
            return [self.im_sai]

        if self.current_frame_index + self.chunk_size < len(self.audio_data):
            chunk = self.audio_data[self.current_frame_index : self.current_frame_index + self.chunk_size]
            self.current_frame_index += self.chunk_size
        else:
            self.current_frame_index = 0
            chunk = self.audio_data[0 : self.chunk_size]
            try: sd.play(self.audio_data, self.sample_rate)
            except: pass

        nap_output = self.processor.process_chunk(chunk)
        sai_output = self.sai_processor.RunSegment(nap_output)
        self.vis.get_vowel_embedding(nap_output)
        self.vis.run_frame(sai_output)
        
        if self.vis.img.shape[1] > 1:
            self.vis.img[:, :-1] = self.vis.img[:, 1:]
            self.vis.draw_column(self.vis.img[:, -1])

        # Vowel Color + Brightness Boost
        vowel_coords = getattr(self.vis, 'vowel_coords', np.array([0.0, 0.0])).flatten()
        vc_x = float(vowel_coords[0]) if len(vowel_coords) > 0 else 0.0
        vc_y = float(vowel_coords[1]) if len(vowel_coords) > 1 else 0.0
        
        r_val = 0.5 - 0.6 * vc_y
        g_val = 0.5 - 0.6 * vc_x
        b_val = 0.35 * (vc_x + vc_y) + 0.4
        tint = np.clip([r_val, g_val, b_val], 0.0, 1.0)
        
        if self.vis.img.ndim == 3: brightness_col = np.mean(self.vis.img[:, -1, :], axis=1)
        else: brightness_col = self.vis.img[:, -1]
             
        target_height = self.rgb_img.shape[0]
        source_height = brightness_col.shape[0]
        if source_height != target_height:
            norm_col = np.interp(np.linspace(0, 1, target_height), np.linspace(0, 1, source_height), brightness_col)
        else: norm_col = brightness_col

        current_max = np.max(self.vis.img) if self.vis.img.size else 1.0
        if current_max < 1e-6: current_max = 1.0
        norm_col = np.clip(norm_col / (current_max * 0.8), 0, 1)
        
        colored_col = (norm_col[:, None] * tint[None, :]) * 2.5 
        
        self.rgb_img[:, :-1, :] = self.rgb_img[:, 1:, :]
        self.rgb_img[:, -1, :] = np.clip(colored_col, 0.0, 1.0)
        
        self.im_sai.set_data(self.rgb_img)
        return [self.im_sai]
    
    def show(self):
        self.ani = animation.FuncAnimation(
            self.fig, self.update_animation, interval=50, blit=False, cache_frame_data=False
        )
        plt.show()

if __name__ == '__main__':
    print("\n" + "="*60)
    print("MANDARIN TONE PERCEPTION — Tone Buttons + Pitchogram")
    print("="*60)
    print("Keys:  1/2/3/4 = pick tone   Space = play   Backspace = undo   p = production mode")
    print("="*60)
    
    app = ToneIntroductionQuizMixed()
    app.show()