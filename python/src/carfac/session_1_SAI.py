import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from matplotlib.patches import FancyBboxPatch
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

# ==========================================
# DESIGN TOKENS
# ==========================================
class Design:
    bg_main = '#FFFFFF'
    bg_dark_card = '#1A1A2E'
    text_main = '#222222'
    text_muted = '#7F8C8D'
    text_mono = '#A0A0B0'
    
    # Distinct tone identity
    tones = {
        1: '#3498DB', # Blue
        2: '#2ECC71', # Green
        3: '#F1C40F', # Amber
        4: '#E74C3C'  # Rose
    }
    tones_light = {
        1: '#EAF2F8', 
        2: '#E9F7EF', 
        3: '#FEF9E7', 
        4: '#FDEDEC'
    }
    
    status = {
        'idle': '#7F8C8D',
        'playing': '#2ECC71',
        'done': '#3498DB'
    }
    
    progress_fill = '#3498DB'
    btn_play = '#3498DB'
    btn_play_hover = '#5DADE2'
    btn_next = '#BDC3C7'
    btn_mode = '#444466'
    correct = '#27AE60'
    incorrect = '#E74C3C'
    
    # Typography
    font_serif = 'Georgia'
    font_mono = 'Courier New'
    font_sans = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']


plt.rcParams['font.family'] = Design.font_serif
plt.rcParams['font.serif'] = [Design.font_serif]
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.sans-serif'] = Design.font_sans
plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# IMPORT THE NEW CONFIGURATION HELPER
# ==========================================
from sai_config import get_sai_params
# ==========================================

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
    TONE_SHAPES = {1: '―', 2: '╱', 3: '∨', 4: '╲'}
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_DIR = os.path.join(SCRIPT_DIR, 'pitchogram_screenshot')

    TONE_IMAGES = {
        1: os.path.join(IMAGE_DIR, 'tone1.png'),
        2: os.path.join(IMAGE_DIR, 'tone2.png'),
        3: os.path.join(IMAGE_DIR, 'tone3.png'),
        4: os.path.join(IMAGE_DIR, 'tone4.png')
    }

    AUDIO_FOLDER = 'mandarin_audio'

    def __init__(self):
        self.script_dir = Path(__file__).parent.resolve()
        self.sample_rate = 16000
        self.chunk_size = 450
        
        self.audio_folder = self._find_folder(self.AUDIO_FOLDER)
        if not self.audio_folder:
            print(f"ERROR: Folder '{self.AUDIO_FOLDER}' not found "
                  f"in {self.script_dir} or its parent.")
        
        all_items = self._scan_folder(self.audio_folder)
        items_one = [it for it in all_items if it['syllables'] == 1]
        items_two = [it for it in all_items if it['syllables'] == 2]

        random.shuffle(items_one)
        random.shuffle(items_two)

        selected_one = items_one[:15]
        selected_two = items_two[:15]
        
        self.vocab_items = selected_one + selected_two
        random.shuffle(self.vocab_items)
            
        print(f"Loaded {len(self.vocab_items)} files "
              f"({len(selected_one)} 1-syllable, {len(selected_two)} 2-syllable) "
              f"from '{self.AUDIO_FOLDER}'.")
        
        self.is_playing = False
        
        self.processor = AudioProcessor(fs=self.sample_rate)
        self.n_channels = self.processor.n_channels
        
        self.sai_params = get_sai_params(self.n_channels, self.chunk_size)
        self.sai_processor = SAIProcessor(self.sai_params)
        self.vis = VisualizationHandler(self.sample_rate, self.sai_params)
        
        self.rgb_img = np.zeros((self.n_channels, 400, 3), dtype=np.float32)

        self.audio_data = None
        self.current_frame_index = 0
        
        self.current_item = None
        self.answered = False
        self.question_count = 0
        self.max_questions = len(self.vocab_items)
        self.results = [] 

        self.selected_tones = []
        
        self.fig = plt.figure(figsize=(11, 10))
        self.fig.patch.set_facecolor(Design.bg_main)
        self.fig.canvas.manager.set_window_title("")       
        
        self.timer_started = False
        self.question_start_time = None
        
        self._setup_interface()

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
        items = []
        if not folder_path:
            return items
        
        for file_path in folder_path.glob('*.wav'):
            try:
                parts = file_path.stem.split('_')
                if len(parts) < 3: continue

                tone = parts[-1]
                chinese = parts[-2]
                tone_digits = ''.join(ch for ch in tone if ch.isdigit())
                if not tone_digits: continue
                syllables = len(tone_digits)

                items.append({
                    "id": file_path.name,
                    "chinese": chinese,
                    "tone": tone_digits,
                    "audio_path": file_path,
                    "syllables": syllables,
                })
            except Exception as e:
                print(f"Skipping {file_path.name}: {e}")
        return items

    # ---------------------------------------------------------------
    # NEW UI DESIGN
    # ---------------------------------------------------------------
    def _setup_interface(self):
        self.ax_ui = self.fig.add_axes([0, 0, 1, 1])
        self.ax_ui.axis('off')

        # --- Header Progress Bar ---
        self.progress_text = self.ax_ui.text(
            0.1, 0.965, '', ha='left', va='center',
            fontsize=16, fontfamily=Design.font_serif, color=Design.text_muted
        )
        self.ax_progress_bg = self.fig.add_axes([0.1, 0.94, 0.8, 0.005])
        self.ax_progress_bg.axis('off')
        self.ax_progress_bg.add_patch(plt.Rectangle((0,0), 1, 1, facecolor='#E0E0E0', transform=self.ax_progress_bg.transAxes))
        self.progress_bar = plt.Rectangle((0,0), 0, 1, facecolor=Design.progress_fill, transform=self.ax_progress_bg.transAxes)
        self.ax_progress_bg.add_patch(self.progress_bar)

        # --- Dark Pitchogram Panel ---
        self.ax_panel = self.fig.add_axes([0.10, 0.55, 0.80, 0.36])
        self.ax_panel.axis('off')
        self.ax_panel.add_patch(plt.Rectangle((0,0), 1, 1, facecolor=Design.bg_dark_card, transform=self.ax_panel.transAxes))

        self.ax_panel.text(0.02, 0.92, 'LIVE DISPLAY', ha='left', va='center',
                           fontsize=12, fontfamily=Design.font_mono, color=Design.text_mono, transform=self.ax_panel.transAxes)
        self.status_pill = self.ax_panel.text(0.20, 0.92, '● idle', ha='left', va='center',
                                              fontsize=12, fontfamily=Design.font_mono, color=Design.status['idle'], transform=self.ax_panel.transAxes)

        # SAI Image inside panel
        self.ax_sai = self.fig.add_axes([0.12, 0.62, 0.76, 0.25])
        self.im_sai = self.ax_sai.imshow(
            self.rgb_img, aspect='auto', origin='upper',
            extent=[0, 11.25, self.processor.n_channels, 0]
        )
        self.ax_sai.set_xticks([])
        self.ax_sai.set_yticks([])
        for spine in self.ax_sai.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#333344')
            spine.set_linewidth(1)

        # --- Prompt ---
        self.prompt_text = self.ax_ui.text(
            0.5, 0.48, 'Identify the tone(s)',
            ha='center', va='center',
            fontsize=26, weight='bold', fontfamily=Design.font_serif, color=Design.text_main
        )

        # --- Distinct Tone Cards (label panel + contour image, session_2 style) ---
        n_buttons = 4
        total_width = 0.16
        button_height = 0.10
        side_margin = 0.08
        gap = (1.0 - 2 * side_margin - n_buttons * total_width) / (n_buttons - 1)
        button_y = 0.33

        self.tone_buttons = {}        # kept for _update_tone_buttons compatibility
        self.tone_button_axes = {}    # kept for _update_tone_buttons compatibility
        self.tone_label_axes = {}
        self.tone_image_axes = {}
        self.tone_image_handles = {}

        for i, tone_num in enumerate([1, 2, 3, 4]):
            x = side_margin + i * (total_width + gap)
            base_color  = Design.tones[tone_num]
            light_color = Design.tones_light[tone_num]

            label_w = total_width * 0.40
            img_w   = total_width * 0.60

            # --- Label panel (replaces old Button widget; still clickable) ---
            ax_lbl = self.fig.add_axes([x, button_y, label_w, button_height])
            ax_lbl.set_xticks([]); ax_lbl.set_yticks([])
            ax_lbl.set_facecolor(light_color)
            for spine in ax_lbl.spines.values():
                spine.set_edgecolor(light_color)
                spine.set_linewidth(0)
            ax_lbl.text(
                0.5, 0.5, str(tone_num), ha='center', va='center',
                fontsize=28, weight='bold',
                fontfamily=Design.font_serif, color=base_color,
                transform=ax_lbl.transAxes
            )
            self.tone_label_axes[tone_num] = ax_lbl
            # Mirror old API so _update_tone_buttons still works
            self.tone_button_axes[tone_num] = ax_lbl

            def make_label_clickable(event, t=tone_num, axis=ax_lbl):
                if event.inaxes == axis:
                    self._on_tone_button(t)
            self.fig.canvas.mpl_connect('button_press_event', make_label_clickable)

            # --- Contour image panel ---
            img_x = x + label_w
            ax_img = self.fig.add_axes([img_x, button_y, img_w, button_height])
            ax_img.set_xticks([]); ax_img.set_yticks([])
            ax_img.patch.set_visible(True)
            ax_img.patch.set_facecolor(light_color)
            for spine in ax_img.spines.values():
                spine.set_edgecolor(light_color)
                spine.set_linewidth(0)

            tone_img = self._load_tone_reference_image(tone_num)
            handle = ax_img.imshow(tone_img, aspect='auto')
            self.tone_image_axes[tone_num] = ax_img
            self.tone_image_handles[tone_num] = handle

            def make_img_clickable(event, t=tone_num, axis=ax_img):
                if event.inaxes == axis:
                    self._on_tone_button(t)
            self.fig.canvas.mpl_connect('button_press_event', make_img_clickable)

        # --- Visual Answer Slots ---
        self.ax_answer_area = self.fig.add_axes([0.3, 0.15, 0.4, 0.12])
        self.ax_answer_area.axis('off')
        
        self.slot1 = FancyBboxPatch((0.1, 0.2), 0.35, 0.6, boxstyle="round,pad=0,rounding_size=0.1", 
                                    facecolor='#EFEFEF', edgecolor='#CCCCCC', lw=2, transform=self.ax_answer_area.transAxes)
        self.ax_answer_area.add_patch(self.slot1)
        self.slot1_text = self.ax_answer_area.text(0.275, 0.5, '', ha='center', va='center', fontsize=26, weight='bold', fontfamily=Design.font_sans[0], color='white', transform=self.ax_answer_area.transAxes)
        
        self.slot_divider = self.ax_answer_area.text(0.5, 0.5, '-', ha='center', va='center', fontsize=24, color='#999999', transform=self.ax_answer_area.transAxes)
        
        self.slot2 = FancyBboxPatch((0.55, 0.2), 0.35, 0.6, boxstyle="round,pad=0,rounding_size=0.1", 
                                    facecolor='#EFEFEF', edgecolor='#CCCCCC', lw=2, transform=self.ax_answer_area.transAxes)
        self.ax_answer_area.add_patch(self.slot2)
        self.slot2_text = self.ax_answer_area.text(0.725, 0.5, '', ha='center', va='center', fontsize=26, weight='bold', fontfamily=Design.font_sans[0], color='white', transform=self.ax_answer_area.transAxes)

        # Inline Feedback Badge
        self.feedback_badge = self.ax_ui.text(0.72, 0.22, '', ha='left', va='center', fontsize=18, fontfamily='Segoe UI Symbol', weight='bold')
        self.reveal_char = self.ax_ui.text(0.72, 0.17, '', ha='left', va='center', fontsize=22, fontfamily=Design.font_sans[0], color=Design.text_main)

        # --- Cleaner Action Row (Bottom Alignment) ---
        self.ax_play_btn = plt.axes([0.24, 0.05, 0.14, 0.045])
        self.btn_play = Button(self.ax_play_btn, '▶ Play', color=Design.btn_play, hovercolor=Design.btn_play_hover)
        self.btn_play.label.set_color('white')
        self.btn_play.label.set_fontfamily(Design.font_sans[0])
        self.btn_play.label.set_weight('bold')
        self.btn_play.label.set_fontsize(16)
        self.btn_play.on_clicked(self._handle_play_click)

        self.ax_next_btn = plt.axes([0.40, 0.05, 0.14, 0.045])
        self.btn_next = Button(self.ax_next_btn, 'Next →', color=Design.btn_next, hovercolor='#95A5A6')
        self.btn_next.label.set_color('#222')
        self.btn_next.label.set_weight('bold')
        self.btn_next.label.set_fontsize(16)
        self.btn_next.label.set_fontfamily(Design.font_sans[0])
        self.btn_next.on_clicked(lambda event: self._next_word())

        self.ax_mode_btn = plt.axes([0.56, 0.05, 0.20, 0.045])
        self.btn_mode = Button(self.ax_mode_btn, 'Production Mode', color=Design.btn_mode, hovercolor='#6666AA')
        self.btn_mode.label.set_color('white')
        self.btn_mode.label.set_weight('bold')
        self.btn_mode.label.set_fontsize(16)
        self.btn_mode.on_clicked(lambda event: self._switch_to_production())

        self._update_tone_buttons()


    def _update_tone_buttons(self):
        """Selected state: fill panel with tone colour; resting state: light tint."""
        for t in [1, 2, 3, 4]:
            ax_lbl = self.tone_label_axes[t]
            ax_img = self.tone_image_axes[t]
            base_color  = Design.tones[t]
            light_color = Design.tones_light[t]

            if t in self.selected_tones:
                ax_lbl.set_facecolor(base_color)
                ax_img.patch.set_facecolor(base_color)
                for txt in ax_lbl.texts:
                    txt.set_color('white')
            else:
                ax_lbl.set_facecolor(light_color)
                ax_img.patch.set_facecolor(light_color)
                for txt in ax_lbl.texts:
                    txt.set_color(base_color)
        self.fig.canvas.draw_idle()


    def _update_answer_display(self):
        if not self.current_item:
            return

        n_syllables = self.current_item.get('syllables', 1)

        if n_syllables == 1:
            self.slot2.set_visible(False)
            self.slot2_text.set_visible(False)
            self.slot_divider.set_visible(False)
            self.slot1.set_bounds(0.325, 0.2, 0.35, 0.6)
            self.slot1_text.set_position((0.5, 0.5))
        else:
            self.slot2.set_visible(True)
            self.slot2_text.set_visible(True)
            self.slot_divider.set_visible(True)
            self.slot1.set_bounds(0.1, 0.2, 0.35, 0.6)
            self.slot1_text.set_position((0.275, 0.5))
            self.slot2.set_bounds(0.55, 0.2, 0.35, 0.6)
            self.slot2_text.set_position((0.725, 0.5))

        def fill_slot(slot, text_obj, idx):
            if idx < len(self.selected_tones):
                t = self.selected_tones[idx]
                slot.set_facecolor(Design.tones[t])
                slot.set_edgecolor(Design.tones[t])
                text_obj.set_text(f"{t} {self.TONE_SHAPES.get(t, '?')}")
            else:
                slot.set_facecolor('#EFEFEF')
                slot.set_edgecolor('#CCCCCC')
                text_obj.set_text('')

        fill_slot(self.slot1, self.slot1_text, 0)
        if n_syllables > 1:
            fill_slot(self.slot2, self.slot2_text, 1)

        self._update_tone_buttons()


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

        self.status_pill.set_text('● done')
        self.status_pill.set_color(Design.status['done'])

        self.reveal_char.set_text(f'{self.current_item["chinese"]}')

        if is_correct:
            self.feedback_badge.set_text('✓ Correct')
            self.feedback_badge.set_color(Design.correct)
            self.feedback_badge.set_fontfamily('Segoe UI Symbol')
        else:
            correct_display = ' · '.join(list(correct_answer))
            self.feedback_badge.set_text(f'✗ Incorrect: {correct_display}')
            self.feedback_badge.set_color(Design.incorrect)
            self.feedback_badge.set_fontfamily('Segoe UI Symbol')

        self.btn_play.label.set_text(r'$\blacktriangleright$ Play')
        self.fig.canvas.draw_idle()


    def _handle_play_click(self, event):
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

                self.btn_play.label.set_text(r'$\blacktriangleright$ Play')
                
                self.status_pill.set_text('● playing')
                self.status_pill.set_color(Design.status['playing'])
                self.fig.canvas.draw_idle()
            except Exception as e:
                print(f"Playback error: {e}")


    def _next_word(self):
        if not self.answered and self.current_item:
            return # Force answering before proceeding
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

        self.feedback_badge.set_text('')
        self.reveal_char.set_text('')
        
        self.current_item = self.vocab_items[self.question_count]

        self.btn_play.label.set_text(r'$\blacktriangleright$ Play')
        
        self.status_pill.set_text('● idle')
        self.status_pill.set_color(Design.status['idle'])

        self._update_answer_display()
        self._update_progress()
        self.fig.canvas.draw_idle()


    def _update_progress(self):
        pct = (self.question_count) / self.max_questions if self.max_questions else 0
        self.progress_bar.set_width(pct)
        
        n_syl = self.current_item.get('syllables', 1) if self.current_item else 1
        progress = f"{self.question_count + 1} / {self.max_questions}"
        if n_syl > 1:
            progress += "   (2 syllables)"
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


    def _switch_to_production(self):
        print("Switching to Production Mode…")
        self.feedback_badge.set_text('Switching...')
        self.feedback_badge.set_color('#E67E22')
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
    
    def _load_tone_reference_image(self, tone_number):
        """Load a tone-reference image; fall back to a drawn contour in the tone colour."""
        candidate_paths = [
            Path(self.TONE_IMAGES[tone_number]),
            Path(self.SCRIPT_DIR) / f"tone_{tone_number}.png",
            Path(self.SCRIPT_DIR) / "tone_images" / f"tone_{tone_number}.png",
            Path(self.SCRIPT_DIR) / "assets" / f"tone_{tone_number}.png",
        ]
        for p in candidate_paths:
            if p.exists():
                try:
                    return plt.imread(str(p))
                except Exception as e:
                    print(f"Could not read tone image {p}: {e}")

        # Fallback: draw contour in the matching tone colour
        H, W = 80, 120
        img = np.ones((H, W, 3), dtype=np.float32)   # white background
        xs = np.linspace(0, 1, W)
        if tone_number == 1:
            ys = np.full_like(xs, 0.2)
        elif tone_number == 2:
            ys = 1.0 - xs * 0.8
        elif tone_number == 3:
            ys = 0.4 + 0.55 * (2 * xs - 1) ** 2
            ys = 1.0 - ys
        elif tone_number == 4:
            ys = 0.2 + xs * 0.8
        else:
            ys = np.full_like(xs, 0.5)
        ys_pix = np.clip((ys * (H - 10) + 5).astype(int), 0, H - 1)
        hex_color = Design.tones.get(tone_number, '#222222').lstrip('#')
        rgb = tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        for x, y in zip(np.arange(W), ys_pix):
            for dy in range(-2, 3):
                yy = np.clip(y + dy, 0, H - 1)
                img[yy, x] = rgb
        return img

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