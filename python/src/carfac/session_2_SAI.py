import sys
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.font_manager as fm
from matplotlib.widgets import Button
from matplotlib.patches import FancyBboxPatch
import threading
import queue
import librosa
import argparse
import os
import sounddevice as sd
import wave
from datetime import datetime
import speech_recognition as sr
import json
import time
import traceback
import subprocess
import random
import csv
from pathlib import Path

# ==========================================
# DESIGN TOKENS  (matched to session_1_SAI.py)
# ==========================================
class Design:
    bg_main = '#FFFFFF'
    bg_dark_card = '#1A1A2E'
    text_main = '#222222'
    text_muted = '#7F8C8D'
    text_mono = '#A0A0B0'

    # Distinct tone identity
    tones = {
        1: '#3498DB',  # Blue
        2: '#2ECC71',  # Green
        3: '#F1C40F',  # Amber
        4: '#E74C3C',  # Rose
    }
    tones_light = {
        1: '#EAF2F8',
        2: '#E9F7EF',
        3: '#FEF9E7',
        4: '#FDEDEC',
    }

    status = {
        'idle':      '#7F8C8D',
        'playing':   '#2ECC71',
        'recording': '#E74C3C',
        'done':      '#3498DB',
    }

    progress_fill   = '#3498DB'
    btn_play        = '#3498DB'
    btn_play_hover  = '#5DADE2'
    btn_stop        = '#E74C3C'
    btn_stop_hover  = '#EC7063'
    btn_record      = '#2ECC71'
    btn_record_hover = '#58D68D'
    btn_recording   = '#E74C3C'
    btn_next        = '#BDC3C7'
    btn_next_hover  = '#95A5A6'
    btn_mode        = '#444466'
    btn_mode_hover  = '#6666AA'
    correct         = '#27AE60'
    incorrect       = '#E74C3C'

    # Typography
    font_serif = 'Georgia'
    font_mono  = 'Courier New'
    font_sans  = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']


plt.rcParams['font.family']      = Design.font_serif
plt.rcParams['font.serif']       = [Design.font_serif]
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.sans-serif']  = Design.font_sans
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# IMPORT CONFIG HELPER
# ==========================================
from sai_config import get_sai_params
# ==========================================

# ---------------- Imports Check ----------------
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

from modules.visualization_handler import VisualizationHandler
from modules.recorder import AudioRecorder

# ---------------- Setup Functions ----------------
def setup_chinese_font():
    """Make sure CJK characters render in matplotlib."""
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'STHeiti', 'Heiti TC',
        'Noto Sans CJK', 'WenQuanYi Micro Hei', 'Arial Unicode MS'
    ]
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            # Don't overwrite the global serif default; just make CJK fall-back available.
            plt.rcParams['font.sans-serif'] = [font_name] + Design.font_sans
            plt.rcParams['axes.unicode_minus'] = False
            print(f"Using CJK fallback font: {font_name}")
            return True
    plt.rcParams['font.sans-serif'] = Design.font_sans
    plt.rcParams['axes.unicode_minus'] = False
    print("Warning: No Chinese font found. Chinese characters may not display correctly.")
    return False

setup_chinese_font()

# ---------------- Helper Classes ----------------

class DummyRecorder:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.start_time = 0
        self.callbacks = []

    def add_audio_callback(self, callback):
        self.callbacks.append(callback)

    def start_recording(self):
        print("--- DUMMY RECORDING STARTED (No Mic Detected) ---")
        self.is_recording = True
        self.start_time = time.time()

    def stop_recording(self):
        print("--- DUMMY RECORDING STOPPED ---")
        if not self.is_recording:
            return None
        self.is_recording = False
        duration = time.time() - self.start_time
        num_samples = int(duration * self.sample_rate)
        fake_audio = np.random.uniform(-0.01, 0.01, num_samples)
        for cb in self.callbacks:
            try:
                cb(fake_audio)
            except Exception as e:
                print(f"Callback error: {e}")
        return fake_audio

class VocabList:
    """Manages the hardcoded vocabulary list for both 1-syllable and 2-syllable words"""
    def __init__(self, root_path):
        self.root_path = Path(root_path)
        self.valid_items_one = []
        self.valid_items_two = []
        self.all_items = []

        # --- 1. Define One-Syllable Words ---
        words_one = [
            {"id": 1,  "chinese": "天", "pinyin": "tiān", "tone": "1", "syllables": 1, "audio": "01_天_1.wav", "folder": "mandarin_audio", "type": "word"},
            {"id": 2,  "chinese": "心", "pinyin": "xīn",  "tone": "1", "syllables": 1, "audio": "02_心_1.wav", "folder": "mandarin_audio", "type": "word"},
            {"id": 3,  "chinese": "车", "pinyin": "chē",  "tone": "1", "syllables": 1, "audio": "03_车_1.wav", "folder": "mandarin_audio", "type": "word"},
            {"id": 4,  "chinese": "糖", "pinyin": "táng", "tone": "2", "syllables": 1, "audio": "04_糖_2.wav", "folder": "mandarin_audio", "type": "word"},
            {"id": 5,  "chinese": "人", "pinyin": "rén",  "tone": "2", "syllables": 1, "audio": "05_人_2.wav", "folder": "mandarin_audio", "type": "word"},
            {"id": 6,  "chinese": "白", "pinyin": "bái",  "tone": "2", "syllables": 1, "audio": "06_白_2.wav", "folder": "mandarin_audio", "type": "word"},
            {"id": 7,  "chinese": "老", "pinyin": "lǎo",  "tone": "3", "syllables": 1, "audio": "07_老_3.wav", "folder": "mandarin_audio", "type": "word"},
            {"id": 8,  "chinese": "火", "pinyin": "huǒ",  "tone": "3", "syllables": 1, "audio": "08_火_3.wav", "folder": "mandarin_audio", "type": "word"},
            {"id": 9,  "chinese": "狗", "pinyin": "gǒu",  "tone": "3", "syllables": 1, "audio": "09_狗_3.wav", "folder": "mandarin_audio", "type": "word"},
            {"id": 10, "chinese": "叫", "pinyin": "jiào", "tone": "4", "syllables": 1, "audio": "10_叫_4.wav", "folder": "mandarin_audio", "type": "word"},
            {"id": 11, "chinese": "骂", "pinyin": "mà",   "tone": "4", "syllables": 1, "audio": "11_骂_4.wav", "folder": "mandarin_audio", "type": "word"},
            {"id": 12, "chinese": "去", "pinyin": "qù",   "tone": "4", "syllables": 1, "audio": "12_去_4.wav", "folder": "mandarin_audio", "type": "word"},
        ]

        # --- 2. Define Two-Syllable Words ---
        words_two = [
            {"id": 201, "chinese": "中国", "pinyin": "zhōngguó", "tone": "1-2", "syllables": 2, "audio": "01_中国_12.wav", "folder": "mandarin_audio", "type": "word"},
            {"id": 202, "chinese": "商店", "pinyin": "shāngdiàn","tone": "1-4", "syllables": 2, "audio": "02_商店_14.wav", "folder": "mandarin_audio", "type": "word"},
            {"id": 203, "chinese": "明天", "pinyin": "míngtiān", "tone": "2-1", "syllables": 2, "audio": "03_明天_21.wav", "folder": "mandarin_audio", "type": "word"},
            {"id": 204, "chinese": "牛奶", "pinyin": "niúnǎi",   "tone": "2-3", "syllables": 2, "audio": "04_牛奶_23.wav", "folder": "mandarin_audio", "type": "word"},
            {"id": 205, "chinese": "学校", "pinyin": "xuéxiào",  "tone": "2-4", "syllables": 2, "audio": "05_学校_24.wav", "folder": "mandarin_audio", "type": "word"},
            {"id": 206, "chinese": "老师", "pinyin": "lǎoshī",   "tone": "3-1", "syllables": 2, "audio": "06_老师_31.wav", "folder": "mandarin_audio", "type": "word"},
            {"id": 207, "chinese": "美国", "pinyin": "měiguó",   "tone": "3-2", "syllables": 2, "audio": "07_美国_32.wav", "folder": "mandarin_audio", "type": "word"},
            {"id": 208, "chinese": "面包", "pinyin": "miànbāo",  "tone": "4-1", "syllables": 2, "audio": "08_面包_41.wav", "folder": "mandarin_audio", "type": "word"},
            {"id": 209, "chinese": "问题", "pinyin": "wèntí",    "tone": "4-2", "syllables": 2, "audio": "09_问题_42.wav", "folder": "mandarin_audio", "type": "word"},
            {"id": 210, "chinese": "电脑", "pinyin": "diànnǎo",  "tone": "4-3", "syllables": 2, "audio": "10_电脑_43.wav", "folder": "mandarin_audio", "type": "word"},
        ]

        # --- 3. Validate Files ---
        print("\nChecking Audio Files...")
        for item in words_one:
            full_path = self.root_path / item['folder'] / item['audio']
            if full_path.exists():
                self.valid_items_one.append(item)
                self.all_items.append(item)
            else:
                print(f"  [Missing 1-Syl] {item['folder']}/{item['audio']}")

        for item in words_two:
            full_path = self.root_path / item['folder'] / item['audio']
            if full_path.exists():
                self.valid_items_two.append(item)
                self.all_items.append(item)
            else:
                print(f"  [Missing 2-Syl] {item['folder']}/{item['audio']}")

        print(f"Loaded: {len(self.valid_items_one)} one-syllable, {len(self.valid_items_two)} two-syllable files.")

class PracticeSession:
    """Manages practice session with mixed folder sources"""
    def __init__(self, practice_set, audio_manager, root_path):
        self.practice_set = practice_set
        self.audio_manager = audio_manager
        self.root_path = Path(root_path)
        self.current_index = 0
        self.all_items = practice_set['words']
        self.total_items = len(self.all_items)

    def get_current_item(self):
        if 0 <= self.current_index < self.total_items:
            return self.all_items[self.current_index]
        return None

    def next_item(self):
        self.current_index = (self.current_index + 1) % self.total_items
        return self.get_current_item()

    def previous_item(self):
        self.current_index = (self.current_index - 1) % self.total_items
        return self.get_current_item()

    def get_audio_for_current(self, voice_type='women'):
        item = self.get_current_item()
        if not item:
            return None, None
        audio_filename = item.get('audio')
        folder_name = item.get('folder', 'mandarin_audio_one_syllable')
        if not audio_filename:
            return None, None
        audio_path = self.root_path / folder_name / audio_filename
        if audio_path.exists():
            return str(audio_path), None
        else:
            print(f"Audio file not found: {audio_path}")
            return None, None

    def get_progress_string(self):
        return f"{self.current_index + 1} / {self.total_items}"

class VoiceSelector:
    def __init__(self, initial_voice='women'):
        self.voices = ['women', 'men']
        self.current_voice = initial_voice
    def toggle(self):
        self.current_voice = self.voices[0] if self.current_voice == self.voices[1] else self.voices[1]
        return self.current_voice
    def get_display_name(self):
        return f'{self.current_voice.capitalize()} Voice'

class AudioManager:
    def __init__(self, base_dir='audio'):
        self.base_dir = base_dir

def get_random_practice_set_from_vocablist(vocab_list):
    """Get a randomised practice set: up to 15 one-syllable + 15 two-syllable items."""
    practice_set = {'words': [], 'sentences': []}

    if len(vocab_list.valid_items_one) >= 15:
        selection_one = random.sample(vocab_list.valid_items_one, 15)
    else:
        selection_one = vocab_list.valid_items_one

    if len(vocab_list.valid_items_two) >= 15:
        selection_two = random.sample(vocab_list.valid_items_two, 15)
    else:
        selection_two = vocab_list.valid_items_two

    combined_words = selection_one + selection_two
    random.shuffle(combined_words)
    practice_set['words'] = combined_words

    print(f"Generated Practice Set: {len(combined_words)} items (Target: 22)")
    return practice_set

# ---------------- Processing Handlers ----------------

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
            return nap * channel_scales
        except Exception:
            return np.zeros((self.n_channels, 0), dtype=np.float32)

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

# ---------------- Main Visualization Class ----------------

class SAIVisualizationWithWav2Vec2:
    TONE_SHAPES = {1: '―', 2: '╱', 3: '∨', 4: '╲'}

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_DIR  = os.path.join(SCRIPT_DIR, 'pitchogram_screenshot')

    TONE_IMAGES = {
        1: os.path.join(IMAGE_DIR, 'tone1.png'),
        2: os.path.join(IMAGE_DIR, 'tone2.png'),
        3: os.path.join(IMAGE_DIR, 'tone3.png'),
        4: os.path.join(IMAGE_DIR, 'tone4.png'),
    }

    def __init__(self, audio_file_path=None, chunk_size=512, sample_rate=16000, sai_width=400,
                 debug=True, playback_speed=1.0, loop_audio=False):

        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sai_width = sai_width
        self.debug = debug
        self.playback_speed = playback_speed
        self.loop_audio = loop_audio
        self.sai_speed = 1.5
        self.sai_file_index = 0.0

        # Reference text and target phonemes
        self.reference_text = None
        self.reference_pronunciation = None
        self.translated_text = None
        self.target_phonemes = "ɕiɛɕiɛ"

        # Audio processors
        self.processor_realtime = AudioProcessor(fs=sample_rate)
        self.processor_file = AudioProcessor(fs=sample_rate)
        self.n_channels = self.processor_realtime.n_channels

        self.sai_params = get_sai_params(self.n_channels, self.chunk_size, smoothing_scale=0.5)

        # SAI processors
        self.sai_realtime = SAIProcessor(self.sai_params)
        self.sai_file = SAIProcessor(self.sai_params)

        # Visualization
        self.vis_realtime = VisualizationHandler(sample_rate, self.sai_params)
        self.vis_file = VisualizationHandler(sample_rate, self.sai_params)

        # Audio setup
        self.audio_queue = queue.Queue(maxsize=50)

        # File processing
        self.audio_file_path = audio_file_path
        self.audio_data = None
        self.current_position = 0
        self.duration = 0
        self.total_samples = 0
        self.loop_count = 0

        # Audio playback
        self.audio_playback_enabled = True
        self.audio_output_stream = None
        self.playback_position = 0.0

        # PyAudio and Threads
        self.p = None
        self.stream = None
        self.running = False

        self.playback_finished_flag = False

        # --- PRACTICE MODE INTEGRATION ---
        self.voice_selector = VoiceSelector()
        self.audio_manager = AudioManager()
        self.practice_session = None

        # Recorder Setup
        self.is_recording_simple = False
        self.vocab_list = None
        self.root_path = None

        # --- ATTRIBUTES FOR SAVING ---
        self.results = []
        self.script_dir = Path(__file__).parent
        self.save_dir = Path("sai_recording")
        self.save_dir.mkdir(exist_ok=True)

        # RGB image buffer for the colourised pitchogram (matches session_1 look)
        self.rgb_img = np.zeros((self.n_channels, self.sai_width, 3), dtype=np.float32)

        print("⚠️ Forcing Dummy Recorder for Lab Computer.")
        self.recorder = DummyRecorder(sample_rate=self.sample_rate)

        self._setup_interface()

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------
    def start(self):
        self.running = True
        webbrowser.open("https://google.github.io/carfac/pitchogram_demo/index.html")
        self._setup_audio_playback()
        self._setup_mic_stream()
        threading.Thread(target=self.process_realtime_audio, daemon=True).start()
        self.ani = animation.FuncAnimation(
            self.fig, self.update_visualization,
            interval=int((self.chunk_size / self.sample_rate) * 1000),
            blit=False,
            cache_frame_data=False
        )
        print("Starting visualization...")
        plt.show()

    def stop(self):
        self.running = False
        if self.audio_output_stream:
            try:
                self.audio_output_stream.stop()
                self.audio_output_stream.close()
            except Exception:
                pass
        if hasattr(self, 'input_stream') and self.input_stream:
            try:
                self.input_stream.stop()
                self.input_stream.close()
            except Exception:
                pass
        try:
            plt.close(self.fig)
        except Exception:
            pass
        print("SAIVisualization stopped.")

    # ---------------------------------------------------------------
    # UI INTERFACE  (re-skinned to match session_1)
    # ---------------------------------------------------------------
    def _setup_interface(self):
        self.fig = plt.figure(figsize=(11, 10))
        self.fig.patch.set_facecolor(Design.bg_main)
        try:
            self.fig.canvas.manager.set_window_title("Mandarin Tone — Production Mode")
        except Exception:
            pass

        self.ax_ui = self.fig.add_axes([0, 0, 1, 1])
        self.ax_ui.axis('off')

        # --- Header Progress Bar ---
        self.progress_text = self.ax_ui.text(
            0.1, 0.965, '', ha='left', va='center',
            fontsize=16, fontfamily=Design.font_serif, color=Design.text_muted
        )
        self.ax_progress_bg = self.fig.add_axes([0.1, 0.94, 0.8, 0.005])
        self.ax_progress_bg.axis('off')
        self.ax_progress_bg.add_patch(plt.Rectangle(
            (0, 0), 1, 1, facecolor='#E0E0E0',
            transform=self.ax_progress_bg.transAxes
        ))
        self.progress_bar = plt.Rectangle(
            (0, 0), 0, 1, facecolor=Design.progress_fill,
            transform=self.ax_progress_bg.transAxes
        )
        self.ax_progress_bg.add_patch(self.progress_bar)

        # --- Dark Pitchogram Panel ---
        self.ax_panel = self.fig.add_axes([0.10, 0.55, 0.80, 0.36])
        self.ax_panel.axis('off')
        self.ax_panel.add_patch(plt.Rectangle(
            (0, 0), 1, 1, facecolor=Design.bg_dark_card,
            transform=self.ax_panel.transAxes
        ))

        self.ax_panel.text(
            0.02, 0.92, 'LIVE DISPLAY', ha='left', va='center',
            fontsize=12, fontfamily=Design.font_mono,
            color=Design.text_mono, transform=self.ax_panel.transAxes
        )
        self.status_pill = self.ax_panel.text(
            0.20, 0.92, '● idle', ha='left', va='center',
            fontsize=12, fontfamily=Design.font_mono,
            color=Design.status['idle'], transform=self.ax_panel.transAxes
        )

        # SAI image inside dark panel
        self.ax_sai = self.fig.add_axes([0.12, 0.62, 0.76, 0.25])
        self.im_file = self.ax_sai.imshow(
            self.rgb_img, aspect='auto', origin='upper',
            extent=[0, 11.25, self.processor_file.n_channels, 0]
        )
        self.ax_sai.set_xticks([])
        self.ax_sai.set_yticks([])
        for spine in self.ax_sai.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#333344')
            spine.set_linewidth(1)

        # --- Prompt ---
        self.prompt_text = self.ax_ui.text(
            0.5, 0.49, 'Speak the word',
            ha='center', va='center',
            fontsize=24, weight='bold',
            fontfamily=Design.font_serif, color=Design.text_main
        )

        # --- Tone-reference cards (light backgrounds, like session_1) ---
        n_buttons = 4
        total_width = 0.16
        button_height = 0.10
        side_margin = 0.08
        gap = (1.0 - 2 * side_margin - n_buttons * total_width) / (n_buttons - 1)
        button_y = 0.33

        self.tone_image_axes = {}
        self.tone_label_axes = {}
        self.tone_image_handles = {}

        for i, tone_num in enumerate([1, 2, 3, 4]):
            x = side_margin + i * (total_width + gap)
            base_color  = Design.tones[tone_num]
            light_color = Design.tones_light[tone_num]

            label_w = total_width * 0.40
            img_w   = total_width * 0.60

            # Tone-number "label" panel (mirrors the button layout from session_1)
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

            # Tone-contour image
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

        # --- Current-item display slots (replaces the old "practice_text") ---
        # Re-uses the slot layout from session_1, but as read-only display of the
        # word being practised rather than user input.
        self.ax_answer_area = self.fig.add_axes([0.3, 0.15, 0.4, 0.12])
        self.ax_answer_area.axis('off')

        self.slot1 = FancyBboxPatch(
            (0.1, 0.2), 0.35, 0.6,
            boxstyle="round,pad=0,rounding_size=0.1",
            facecolor='#EFEFEF', edgecolor='#CCCCCC', lw=2,
            transform=self.ax_answer_area.transAxes
        )
        self.ax_answer_area.add_patch(self.slot1)
        # Change this (around line 508):
        self.slot1_text = self.ax_answer_area.text(
            0.275, 0.5, '', ha='center', va='center',
            fontsize=26, weight='bold',
            fontfamily=Design.font_sans[0],  # <--- Change from Design.font_serif
            color='white',
            transform=self.ax_answer_area.transAxes
        )

        self.slot_divider = self.ax_answer_area.text(
            0.5, 0.5, '-', ha='center', va='center',
            fontsize=24, color='#999999',
            transform=self.ax_answer_area.transAxes
        )

        self.slot2 = FancyBboxPatch(
            (0.55, 0.2), 0.35, 0.6,
            boxstyle="round,pad=0,rounding_size=0.1",
            facecolor='#EFEFEF', edgecolor='#CCCCCC', lw=2,
            transform=self.ax_answer_area.transAxes
        )
        self.ax_answer_area.add_patch(self.slot2)
        self.slot2_text = self.ax_answer_area.text(
            0.725, 0.5, '', ha='center', va='center',
            fontsize=26, weight='bold',
            fontfamily=Design.font_sans[0],  # <--- Change from Design.font_serif
            color='white',
            transform=self.ax_answer_area.transAxes
        )

        # Chinese character + pinyin reveal (under the slots)
        self.practice_text = self.ax_ui.text(
            0.5, 0.13, '', ha='center', va='center',
            fontsize=22, fontfamily=Design.font_sans[0],
            color=Design.text_main
        )

        # Status text (kept for blit compatibility — invisible because the
        # status pill in the dark panel already fulfils this role)
        self.status_text = self.ax_ui.text(
            0, 0, '', alpha=0
        )

        # --- Action button row — evenly spaced, fits narrow/dual-screen ---
        # [  ▶ Play Reference  ] [   Next →   ] [  Perception Mode  ]
        btn_h   = 0.050
        btn_y   = 0.028
        margin  = 0.05          # left/right page margin
        gap     = 0.015         # gap between buttons
        n_btns  = 3
        btn_w   = (1.0 - 2 * margin - (n_btns - 1) * gap) / n_btns  # ≈ 0.272 each

        x1 = margin
        x2 = x1 + btn_w + gap
        x3 = x2 + btn_w + gap

        self.ax_play_btn = plt.axes([x1, btn_y, btn_w, btn_h])
        self.btn_playback = Button(
            self.ax_play_btn, r'$\blacktriangleright$ Play',
            color=Design.btn_play, hovercolor=Design.btn_play_hover
        )
        self.btn_playback.label.set_color('white')
        self.btn_playback.label.set_fontfamily(Design.font_sans[0])
        self.btn_playback.label.set_weight('bold')
        self.btn_playback.label.set_fontsize(13)
        self.btn_playback.on_clicked(self.toggle_playback)

        self.ax_next_btn = plt.axes([x2, btn_y, btn_w, btn_h])
        self.btn_next = Button(
            self.ax_next_btn, 'Next →',
            color=Design.btn_next, hovercolor=Design.btn_next_hover
        )
        self.btn_next.label.set_color('#222')
        self.btn_next.label.set_weight('bold')
        self.btn_next.label.set_fontsize(13)
        self.btn_next.label.set_fontfamily(Design.font_sans[0])
        self.btn_next.on_clicked(self.next_item)

        self.ax_perception_btn = plt.axes([x3, btn_y, btn_w, btn_h])
        self.btn_perception = Button(
            self.ax_perception_btn, 'Perception Mode',
            color=Design.btn_mode, hovercolor=Design.btn_mode_hover
        )
        self.btn_perception.label.set_color('white')
        self.btn_perception.label.set_weight('bold')
        self.btn_perception.label.set_fontsize(13)
        self.btn_perception.on_clicked(self.switch_to_perception)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    # ---------------------------------------------------------------
    # Status / display helpers
    # ---------------------------------------------------------------
    def _set_status(self, text, color):
        """Update the status pill in the dark panel."""
        if hasattr(self, 'status_pill'):
            self.status_pill.set_text(f'● {text}')
            self.status_pill.set_color(color)
            self.fig.canvas.draw_idle()

    def _update_progress(self):
        if not self.practice_session:
            return
        total = self.practice_session.total_items
        idx = self.practice_session.current_index
        pct = idx / total if total else 0
        self.progress_bar.set_width(pct)

        progress = f"{idx + 1} / {total}"
        item = self.practice_session.get_current_item()
        if item and item.get('syllables', 1) > 1:
            progress += "   (2 syllables)"
        self.progress_text.set_text(progress)

    def _update_item_display(self, item):
        """Fill the answer slots with the current item's tones (read-only)."""
        if not item:
            self.slot1.set_visible(False)
            self.slot2.set_visible(False)
            self.slot1_text.set_visible(False)
            self.slot2_text.set_visible(False)
            self.slot_divider.set_visible(False)
            return

        n_syllables = item.get('syllables', 1)

        # Parse tone(s)
        tone_field = str(item.get('tone', ''))
        tones = []
        for part in tone_field.replace(',', '-').split('-'):
            part = part.strip()
            if part.isdigit():
                tones.append(int(part))

        if n_syllables == 1:
            self.slot2.set_visible(False)
            self.slot2_text.set_visible(False)
            self.slot_divider.set_visible(False)
            self.slot1.set_visible(True)
            self.slot1_text.set_visible(True)
            self.slot1.set_bounds(0.325, 0.2, 0.35, 0.6)
            self.slot1_text.set_position((0.5, 0.5))
        else:
            self.slot1.set_visible(True)
            self.slot1_text.set_visible(True)
            self.slot2.set_visible(True)
            self.slot2_text.set_visible(True)
            self.slot_divider.set_visible(True)
            self.slot1.set_bounds(0.1, 0.2, 0.35, 0.6)
            self.slot1_text.set_position((0.275, 0.5))
            self.slot2.set_bounds(0.55, 0.2, 0.35, 0.6)
            self.slot2_text.set_position((0.725, 0.5))

        def fill_slot(slot, text_obj, idx):
            if idx < len(tones):
                t = tones[idx]
                color = Design.tones.get(t, Design.text_muted)
                slot.set_facecolor(color)
                slot.set_edgecolor(color)
                text_obj.set_text(f"{t} {self.TONE_SHAPES.get(t, '?')}")
            else:
                slot.set_facecolor('#EFEFEF')
                slot.set_edgecolor('#CCCCCC')
                text_obj.set_text('')

        fill_slot(self.slot1, self.slot1_text, 0)
        if n_syllables > 1:
            fill_slot(self.slot2, self.slot2_text, 1)

        # Show character and pinyin
        chinese = item.get('chinese', '')
        pinyin  = item.get('pinyin', '')
        self.practice_text.set_text(f"{chinese}   ({pinyin})")

        # Highlight tone reference cards
        self._highlight_tone_images_for_item(item)

    def _highlight_tone_images_for_item(self, item):
        """Visually highlight the tone-reference cards for this item's tones."""
        tone_field = str(item.get('tone', ''))
        active_tones = set()
        for part in tone_field.replace(',', '-').split('-'):
            part = part.strip()
            if part.isdigit():
                n = int(part)
                if 1 <= n <= 4:
                    active_tones.add(n)

        for tone_num in [1, 2, 3, 4]:
            ax_img = self.tone_image_axes.get(tone_num)
            ax_lbl = self.tone_label_axes.get(tone_num)
            if ax_img is None or ax_lbl is None:
                continue
            base_color  = Design.tones[tone_num]
            light_color = Design.tones_light[tone_num]
            is_active = tone_num in active_tones

            if is_active:
                # Filled (active) state — mirrors session_1's "selected" look
                ax_lbl.set_facecolor(base_color)
                ax_img.patch.set_facecolor(base_color)
                # Repaint the number in white
                for txt in ax_lbl.texts:
                    txt.set_color('white')
            else:
                # Resting state
                ax_lbl.set_facecolor(light_color)
                ax_img.patch.set_facecolor(light_color)
                for txt in ax_lbl.texts:
                    txt.set_color(base_color)

    # ---------------------------------------------------------------
    # Practice item loading
    # ---------------------------------------------------------------
    def _load_practice_item(self, item):
        if not item or not self.practice_session:
            return

        reference_pronunciation = item.get('pinyin', item.get('chinese'))
        translation = item.get('english', '')
        target_phonemes = item.get('phonemes', 'placeholder')

        self.set_reference_text(target_phonemes, reference_pronunciation, translation)

        # Update visible slots and pitchogram clear
        self._update_item_display(item)
        self._update_progress()

        audio_path, _ = self.practice_session.get_audio_for_current(self.voice_selector.current_voice)

        if audio_path and os.path.exists(audio_path):
            audio_data, original_sr = librosa.load(audio_path, sr=None)
            if original_sr != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=self.sample_rate)
            self.audio_data = audio_data
            self.total_samples = len(self.audio_data)
            self.current_position = 0

            self.vis_file.img[:] = 0
            self.rgb_img[:] = 0
            self.im_file.set_data(self.rgb_img)

        self._set_status('idle', Design.status['idle'])
        self.fig.canvas.draw_idle()

    def clear_phoneme_feedback(self, event=None):
        self._set_status('idle', Design.status['idle'])

    # ---------------------------------------------------------------
    # Speed / key handlers
    # ---------------------------------------------------------------
    def decrease_sai_speed(self, event=None):
        self.sai_speed = max(0.1, self.sai_speed - 0.25)
    def increase_sai_speed(self, event=None):
        self.sai_speed = min(5.0, self.sai_speed + 0.25)
    def decrease_audio_speed(self, event=None):
        self.playback_speed = max(0.25, self.playback_speed - 0.25)
    def increase_audio_speed(self, event=None):
        self.playback_speed = min(5.0, self.playback_speed + 0.25)

    def on_key_press(self, event):
        if event.key in ('up', '+'):
            self.increase_sai_speed()
        elif event.key in ('down', '-'):
            self.decrease_sai_speed()
        elif event.key == 'right':
            if self.practice_session:
                self.next_item()
            else:
                self.increase_audio_speed()
        elif event.key == 'left':
            self.decrease_audio_speed()
        elif event.key == 'r':
            self.toggle_record()
        elif event.key == ' ':
            self.toggle_playback()
        elif event.key == 'c':
            self.clear_phoneme_feedback()

    # ---------------------------------------------------------------
    # Audio loading / playback
    # ---------------------------------------------------------------
    def _load_audio_file(self):
        print(f"Loading audio file: {self.audio_file_path}")
        self.audio_data, original_sr = librosa.load(self.audio_file_path, sr=None)
        if original_sr != self.sample_rate:
            self.audio_data = librosa.resample(self.audio_data, orig_sr=original_sr, target_sr=self.sample_rate)

        if np.max(np.abs(self.audio_data)) > 0:
            self.audio_data = self.audio_data / np.max(np.abs(self.audio_data)) * 0.9

        self.total_samples = len(self.audio_data)
        self.duration = self.total_samples / self.sample_rate
        if self.audio_playback_enabled:
            self._setup_audio_playback()

    def set_reference_text(self, phonemes, pronunciation, translation):
        self.reference_text = phonemes.strip()
        self.reference_pronunciation = pronunciation
        self.translated_text = translation.strip()

    def _setup_audio_playback(self):
        try:
            self.audio_output_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=self._audio_playback_callback
            )
        except Exception as e:
            print(f"Failed to create audio playback: {e}")
            self.audio_playback_enabled = False

    def _on_playback_finished(self):
        self.playback_finished_flag = True

    def _play_audio_file(self, audio_data, sample_rate):
        if self.audio_output_stream and self.audio_output_stream.active:
            self.audio_output_stream.stop()

        self.playback_position = 0.0
        self.loop_count = 0

        try:
            self.audio_output_stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=self._audio_playback_callback,
                finished_callback=self._on_playback_finished
            )
            self.audio_output_stream.start()
            print(f"Playing reference audio ({self.duration:.1f}s)")
        except Exception as e:
            print(f"Failed to create audio playback: {e}")
            self.audio_playback_enabled = False

    def _audio_playback_callback(self, outdata, frames, time, status):
        try:
            if self.audio_data is not None:
                start_pos = int(self.playback_position)
                speed_factor = self.playback_speed

                chunk_indices = np.arange(frames) * speed_factor
                chunk_indices = chunk_indices.astype(int) + start_pos

                if np.any(chunk_indices >= self.total_samples):
                    valid_mask = chunk_indices < self.total_samples
                    chunk = np.zeros(frames, dtype=np.float32)
                    valid_indices = chunk_indices[valid_mask]
                    if len(valid_indices) > 0:
                        chunk[:len(valid_indices)] = self.audio_data[valid_indices]
                    outdata[:len(chunk), 0] = chunk
                    raise sd.CallbackStop
                else:
                    chunk = self.audio_data[chunk_indices]
                    outdata[:len(chunk), 0] = chunk
                    self.playback_position += int(frames * speed_factor)
            else:
                outdata.fill(0)
                raise sd.CallbackStop
        except sd.CallbackStop:
            raise sd.CallbackStop
        except Exception as e:
            print(f"Playback error: {e}")
            outdata.fill(0)
            raise sd.CallbackStop

    def get_next_file_chunk(self):
        if self.audio_data is None or self.total_samples == 0:
            return None, -1
        if self.current_position >= self.total_samples:
            if self.loop_audio:
                self.current_position = 0
                self.loop_count += 1
            else:
                return None, -1
        end_position = min(self.current_position + self.chunk_size, self.total_samples)
        chunk = self.audio_data[self.current_position:end_position]
        if len(chunk) < self.chunk_size:
            chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), 'constant')
        chunk_index = self.current_position
        self.current_position = end_position
        return chunk.astype(np.float32), chunk_index

    def _setup_mic_stream(self):
        try:
            self.input_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.chunk_size,
                callback=self._audio_input_callback,
                dtype=np.float32
            )
            self.input_stream.start()
            print("Microphone stream started successfully via SoundDevice.")
        except Exception as e:
            print(f"Error starting microphone: {e}")

    def _audio_input_callback(self, indata, frames, time_info, status):
        try:
            audio_data = indata.copy().flatten()
            if not self.audio_queue.full():
                self.audio_queue.put(audio_data)
        except Exception:
            pass

    def process_realtime_audio(self):
        while self.running:
            try:
                self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            # Real-time stream is drained — no live pitchogram for mic input here.

    # ---------------------------------------------------------------
    # Animation update — colourised pitchogram (matches session_1 look)
    # ---------------------------------------------------------------
    def update_visualization(self, frame):
        # Reset playback button when audio finishes
        if self.playback_finished_flag:
            self.playback_finished_flag = False
            self.btn_playback.label.set_text(r'$\blacktriangleright$ Play')
            self.btn_playback.color = Design.btn_play
            self.btn_playback.hovercolor = Design.btn_play_hover
            self._set_status('done', Design.status['done'])
            self.fig.canvas.draw_idle()

        try:
            if self.audio_data is not None:
                chunk, chunk_index = self.get_next_file_chunk()
                if chunk is not None and chunk_index >= 0:
                    nap_output = self.processor_file.process_chunk(chunk)
                    sai_output = self.sai_file.RunSegment(nap_output)
                    self.vis_file.get_vowel_embedding(nap_output)
                    self.vis_file.run_frame(sai_output)

                    self.sai_file_index += self.sai_speed
                    if self.sai_file_index >= 1.0:
                        steps = int(self.sai_file_index)
                        self.sai_file_index -= steps
                        for _ in range(min(steps, 3)):
                            if self.vis_file.img.shape[1] > 1:
                                self.vis_file.img[:, :-1] = self.vis_file.img[:, 1:]
                                self.vis_file.draw_column(self.vis_file.img[:, -1])

                    # --- Tint by vowel embedding (session_1 style) ---
                    vowel_coords = getattr(self.vis_file, 'vowel_coords', np.array([0.0, 0.0])).flatten()
                    vc_x = float(vowel_coords[0]) if len(vowel_coords) > 0 else 0.0
                    vc_y = float(vowel_coords[1]) if len(vowel_coords) > 1 else 0.0

                    r_val = 0.5 - 0.6 * vc_y
                    g_val = 0.5 - 0.6 * vc_x
                    b_val = 0.35 * (vc_x + vc_y) + 0.4
                    tint = np.clip([r_val, g_val, b_val], 0.0, 1.0)

                    if self.vis_file.img.ndim == 3:
                        brightness_col = np.mean(self.vis_file.img[:, -1, :], axis=1)
                    else:
                        brightness_col = self.vis_file.img[:, -1]

                    target_height = self.rgb_img.shape[0]
                    source_height = brightness_col.shape[0]
                    if source_height != target_height:
                        norm_col = np.interp(
                            np.linspace(0, 1, target_height),
                            np.linspace(0, 1, source_height),
                            brightness_col
                        )
                    else:
                        norm_col = brightness_col

                    current_max = np.max(self.vis_file.img) if self.vis_file.img.size else 1.0
                    if current_max < 1e-6:
                        current_max = 1.0
                    norm_col = np.clip(norm_col / (current_max * 0.8), 0, 1)

                    colored_col = (norm_col[:, None] * tint[None, :]) * 2.5

                    self.rgb_img[:, :-1, :] = self.rgb_img[:, 1:, :]
                    self.rgb_img[:, -1, :]  = np.clip(colored_col, 0.0, 1.0)

                    self.im_file.set_data(self.rgb_img)
        except Exception:
            pass

        return [self.im_file]

    # ---------------------------------------------------------------
    # Buttons / actions
    # ---------------------------------------------------------------
    def toggle_record(self, event=None):
        try:
            if not self.is_recording_simple:
                print("Attempting to start recording...")
                self.is_recording_simple = True

                self.btn_record.label.set_text('■ Stop & Save')
                self.btn_record.color = Design.btn_recording
                self.btn_record.hovercolor = '#EC7063'

                self.recorder.start_recording()
                self._set_status('recording', Design.status['recording'])
                self.fig.canvas.draw_idle()
            else:
                self.is_recording_simple = False

                self.btn_record.label.set_text('● Start Record')
                self.btn_record.color = Design.btn_record
                self.btn_record.hovercolor = Design.btn_record_hover

                recorded_audio = self.recorder.stop_recording()
                if recorded_audio is not None:
                    self.save_recorded_audio(recorded_audio)
                else:
                    print("Warning: No audio data received from recorder.")
                self.fig.canvas.draw_idle()

        except Exception as e:
            print(f"\n⚠️ RECORDING ERROR: {e}")
            self.is_recording_simple = False
            self.btn_record.label.set_text('● Start Record')
            self.btn_record.color = Design.btn_record
            self._set_status('Microphone Error', Design.incorrect)
            self.fig.canvas.draw_idle()

    def next_item(self, event=None):
        if self.practice_session:
            if self.practice_session.current_index >= self.practice_session.total_items - 1:
                self._set_status(
                    f"✓ Practice Set Complete ({self.practice_session.total_items}/{self.practice_session.total_items})",
                    Design.correct,
                )
                if self.results:
                    self._save_results_to_csv()
                plt.close(self.fig)
                self._launch_next_script()
            else:
                item = self.practice_session.next_item()
                self._load_practice_item(item)
        self.fig.canvas.draw_idle()

    def toggle_playback(self, event=None):
        if self.audio_output_stream and self.audio_output_stream.active:
            # --- STOPPING MANUALLY ---
            self.audio_output_stream.stop()
            self.btn_playback.label.set_text(r'$\blacktriangleright$ Play')
            self.btn_playback.color = Design.btn_play
            self.btn_playback.hovercolor = Design.btn_play_hover
            self._set_status('idle', Design.status['idle'])
        else:
            # --- STARTING PLAYBACK ---
            current_item = self.practice_session.get_current_item() if self.practice_session else None
            if current_item:
                audio_path, _ = self.practice_session.get_audio_for_current(self.voice_selector.current_voice)
                if audio_path and os.path.exists(audio_path):
                    audio_data, original_sr = librosa.load(audio_path, sr=None)
                    if original_sr != self.sample_rate:
                        audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=self.sample_rate)

                    self.playback_position = 0.0
                    self._play_audio_file(audio_data, self.sample_rate)

                    self.btn_playback.label.set_text('■ Stop Ref')
                    self.btn_playback.color = Design.btn_stop
                    self.btn_playback.hovercolor = Design.btn_stop_hover
                    self._set_status('playing', Design.status['playing'])

        self.fig.canvas.draw_idle()

    def _launch_next_script(self):
        target_file = "tone_production_SAI_two_syllable.py"
        current_dir = Path(__file__).parent
        next_script = current_dir / target_file
        if not next_script.exists():
            next_script = current_dir.parent / "session_1_tone_recognition" / target_file

        if next_script.exists():
            print(f"🚀 Launching next script: {next_script}")
            subprocess.Popen([sys.executable, str(next_script)])
        else:
            print(f"⚠️ Could not find next script: {target_file}")

    def switch_to_perception(self, event=None):
        """Close production prototype and launch perception prototype."""
        target_file = "session_1_SAI.py"
        current_dir = Path(__file__).parent
        perception_script = current_dir / target_file
        if not perception_script.exists():
            perception_script = current_dir.parent / "session_1_tone_recognition" / target_file
        if not perception_script.exists():
            perception_script = current_dir.parent / target_file

        if perception_script.exists():
            print(f"🔀 Switching to Perception prototype: {perception_script}")
            self._set_status('Switching...', '#E67E22')
            self.fig.canvas.draw_idle()
            plt.pause(0.4)

            if self.results:
                self._save_results_to_csv()
            try:
                subprocess.Popen([sys.executable, str(perception_script)])
            except Exception as e:
                print(f"⚠️ Failed to launch perception script: {e}")
                return
            self.running = False
            try:
                plt.close(self.fig)
            except Exception:
                pass
        else:
            print(f"⚠️ Could not find perception script: {target_file}")
            self._set_status(f'Perception script not found', Design.incorrect)

    # ---------------------------------------------------------------
    # Tone-reference image loader (re-used from old code)
    # ---------------------------------------------------------------
    def _load_tone_reference_image(self, tone_number):
        """Load a tone reference contour image, with a clean drawn fallback."""
        candidate_paths = [
            Path(self.TONE_IMAGES[tone_number]),
            self.script_dir / f"tone_{tone_number}.png",
            self.script_dir / "tone_images" / f"tone_{tone_number}.png",
            self.script_dir / "assets" / f"tone_{tone_number}.png",
            self.script_dir / "pitchogram_screenshot" / f"tone{tone_number}.png",
        ]
        for p in candidate_paths:
            if p.exists():
                try:
                    return plt.imread(str(p))
                except Exception as e:
                    print(f"Could not read tone image {p}: {e}")

        # Fallback: draw a simple tone contour
        H, W = 80, 120
        img = np.ones((H, W, 3), dtype=np.float32)  # white background
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
        # Draw the curve in the matching tone colour, thicker for clarity
        hex_color = Design.tones.get(tone_number, '#222222').lstrip('#')
        rgb = tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        for x, y in zip(np.arange(W), ys_pix):
            for dy in range(-2, 3):
                yy = np.clip(y + dy, 0, H - 1)
                img[yy, x] = rgb
        return img

# ---------------- Main Execution ----------------

def main():
    parser = argparse.ArgumentParser(description="SAI Visualization and Mandarin Pronunciation Practice Tool.")
    parser.add_argument("--word", type=str, help="Specify a single Mandarin word for practice.")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    vocab_list = VocabList(root_path=str(script_dir))

    if len(vocab_list.all_items) == 0:
        print("❌ No audio files found in subfolders. Check your directory structure.")

    practice_set = None
    audio_file_path = None
    word_info = None

    if args.word:
        word_info = next(
            (item for item in vocab_list.all_items
             if item.get('type') == 'word' and item.get('chinese') == args.word),
            None
        )
        if word_info:
            audio_file_path = str(script_dir / word_info['folder'] / word_info['audio'])
            print(f"Single word mode: {word_info['chinese']} ({word_info['pinyin']})")
        else:
            print(f"Word '{args.word}' not found in vocabulary.")
            return 1
    else:
        print("--- Starting in Practice Mode ---")
        practice_set = get_random_practice_set_from_vocablist(vocab_list)
        if not practice_set['words']:
            print("❌ No practice items generated.")
            return 1
        word_info = practice_set['words'][0]

    try:
        sai_vis = SAIVisualizationWithWav2Vec2(
            audio_file_path=audio_file_path,
            playback_speed=1.0,
            loop_audio=False
        )

        sai_vis.vocab_list = vocab_list
        sai_vis.root_path = script_dir

        if practice_set:
            practice_session = PracticeSession(practice_set, sai_vis.audio_manager, root_path=str(script_dir))
            sai_vis.practice_session = practice_session
            sai_vis._load_practice_item(practice_session.get_current_item())
        elif word_info:
            reference_pronunciation = word_info.get('pinyin', word_info.get('chinese'))
            translation = word_info.get('english', '')
            target_phonemes = word_info.get('phonemes', 'placeholder')
            sai_vis.set_reference_text(target_phonemes, reference_pronunciation, translation)
            sai_vis._load_audio_file()

        sai_vis.start()

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error starting visualization: {e}")
        traceback.print_exc()
        return 1
    finally:
        if 'sai_vis' in locals():
            sai_vis.stop()
        print("✅ Visualization stopped cleanly")

    return 0

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("MANDARIN TONE PRODUCTION — Pitchogram + Recording")
    print("=" * 60)
    print("Keys:  Space = play   r = record   → = next   c = clear")
    print("=" * 60)
    sys.exit(main())