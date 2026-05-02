import sys
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
import json
import time
import traceback
import subprocess
import random
import csv
from pathlib import Path

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
    """Setup matplotlib to display Chinese characters"""
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'STHeiti', 'Heiti TC',
        'Noto Sans CJK', 'WenQuanYi Micro Hei', 'Arial Unicode MS'
    ]
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"Using font: {font_name}")
            return True
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("Warning: No Chinese font found.")
    return False


def get_font_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(script_dir, "DoulosSIL-Regular.ttf"),
        os.path.join(script_dir, "fonts", "DoulosSIL-Regular.ttf"),
        os.path.join(script_dir, "DoulosSIL-7.000", "DoulosSIL-Regular.ttf"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


setup_chinese_font()
font_path = get_font_path()
if font_path:
    font_prop = fm.FontProperties(fname=font_path, size=16)
else:
    font_prop = fm.FontProperties(family='Times New Roman', size=16)


# ---------------- Helper Classes ----------------

class VocabList:
    """Manages the hardcoded vocabulary list for both 1-syllable and 2-syllable words"""

    def __init__(self, root_path):
        self.root_path = Path(root_path)
        self.valid_items_one = []
        self.valid_items_two = []
        self.all_items = []

        words_one = [
            {"id": 1,  "chinese": "天", "pinyin": "tiān", "tone": "1", "syllables": 1, "audio": "01_天_1.wav", "folder": "mandarin_audio_one_syllable", "type": "word"},
            {"id": 2,  "chinese": "心", "pinyin": "xīn",  "tone": "1", "syllables": 1, "audio": "02_心_1.wav", "folder": "mandarin_audio_one_syllable", "type": "word"},
            {"id": 3,  "chinese": "车", "pinyin": "chē",  "tone": "1", "syllables": 1, "audio": "03_车_1.wav", "folder": "mandarin_audio_one_syllable", "type": "word"},
            {"id": 4,  "chinese": "学", "pinyin": "xué",  "tone": "2", "syllables": 1, "audio": "04_学_2.wav", "folder": "mandarin_audio_one_syllable", "type": "word"},
            {"id": 5,  "chinese": "人", "pinyin": "rén",  "tone": "2", "syllables": 1, "audio": "05_人_2.wav", "folder": "mandarin_audio_one_syllable", "type": "word"},
            {"id": 6,  "chinese": "白", "pinyin": "bái",  "tone": "2", "syllables": 1, "audio": "06_白_2.wav", "folder": "mandarin_audio_one_syllable", "type": "word"},
            {"id": 7,  "chinese": "老", "pinyin": "lǎo",  "tone": "3", "syllables": 1, "audio": "07_老_3.wav", "folder": "mandarin_audio_one_syllable", "type": "word"},
            {"id": 8,  "chinese": "火", "pinyin": "huǒ",  "tone": "3", "syllables": 1, "audio": "08_火_3.wav", "folder": "mandarin_audio_one_syllable", "type": "word"},
            {"id": 9,  "chinese": "狗", "pinyin": "gǒu",  "tone": "3", "syllables": 1, "audio": "09_狗_3.wav", "folder": "mandarin_audio_one_syllable", "type": "word"},
            {"id": 10, "chinese": "叫", "pinyin": "jiào", "tone": "4", "syllables": 1, "audio": "10_叫_4.wav", "folder": "mandarin_audio_one_syllable", "type": "word"},
            {"id": 11, "chinese": "骂", "pinyin": "mà",   "tone": "4", "syllables": 1, "audio": "11_骂_4.wav", "folder": "mandarin_audio_one_syllable", "type": "word"},
            {"id": 12, "chinese": "去", "pinyin": "qù",   "tone": "4", "syllables": 1, "audio": "12_去_4.wav", "folder": "mandarin_audio_one_syllable", "type": "word"},
        ]

        words_two = [
            {"id": 201, "chinese": "中国", "pinyin": "zhōngguó", "tone": "1-2", "syllables": 2, "audio": "01_中国_12.wav", "folder": "mandarin_audio_two_syllable", "type": "word"},
            {"id": 202, "chinese": "商店", "pinyin": "shāngdiàn", "tone": "1-4", "syllables": 2, "audio": "02_商店_14.wav", "folder": "mandarin_audio_two_syllable", "type": "word"},
            {"id": 203, "chinese": "明天", "pinyin": "míngtiān", "tone": "2-1", "syllables": 2, "audio": "03_明天_21.wav", "folder": "mandarin_audio_two_syllable", "type": "word"},
            {"id": 204, "chinese": "牛奶", "pinyin": "niúnǎi",   "tone": "2-3", "syllables": 2, "audio": "04_牛奶_23.wav", "folder": "mandarin_audio_two_syllable", "type": "word"},
            {"id": 205, "chinese": "学校", "pinyin": "xuéxiào",  "tone": "2-4", "syllables": 2, "audio": "05_学校_24.wav", "folder": "mandarin_audio_two_syllable", "type": "word"},
            {"id": 206, "chinese": "老师", "pinyin": "lǎoshī",   "tone": "3-1", "syllables": 2, "audio": "06_老师_31.wav", "folder": "mandarin_audio_two_syllable", "type": "word"},
            {"id": 207, "chinese": "美国", "pinyin": "měiguó",   "tone": "3-2", "syllables": 2, "audio": "07_美国_32.wav", "folder": "mandarin_audio_two_syllable", "type": "word"},
            {"id": 208, "chinese": "面包", "pinyin": "miànbāo",  "tone": "4-1", "syllables": 2, "audio": "08_面包_41.wav", "folder": "mandarin_audio_two_syllable", "type": "word"},
            {"id": 209, "chinese": "问题", "pinyin": "wèntí",    "tone": "4-2", "syllables": 2, "audio": "09_问题_42.wav", "folder": "mandarin_audio_two_syllable", "type": "word"},
            {"id": 210, "chinese": "电脑", "pinyin": "diànnǎo",  "tone": "4-3", "syllables": 2, "audio": "10_电脑_43.wav", "folder": "mandarin_audio_two_syllable", "type": "word"},
        ]

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

    def __init__(self, practice_set, root_path):
        self.practice_set = practice_set
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

    def get_audio_for_current(self):
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


def get_random_practice_set_from_vocablist(vocab_list):
    """Get exactly 3 random one-syllable and 3 random two-syllable words."""
    practice_set = {'words': [], 'sentences': []}

    if len(vocab_list.valid_items_one) >= 3:
        selection_one = random.sample(vocab_list.valid_items_one, 3)
    else:
        selection_one = vocab_list.valid_items_one

    if len(vocab_list.valid_items_two) >= 3:
        selection_two = random.sample(vocab_list.valid_items_two, 3)
    else:
        selection_two = vocab_list.valid_items_two

    combined_words = selection_one + selection_two
    random.shuffle(combined_words)
    practice_set['words'] = combined_words

    print(f"Generated Practice Set: {len(combined_words)} items")
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
            nap = nap * channel_scales
            return nap
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

class TonePerceptionApp:
    """Perception Mode: Listen to reference audio and identify tone(s)."""

    def __init__(self, chunk_size=512, sample_rate=16000, sai_width=400):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sai_width = sai_width
        self.sai_speed = 1.5
        self.sai_file_index = 0.0
        self.playback_speed = 1.0

        # Audio processors
        self.processor_file = AudioProcessor(fs=sample_rate)
        self.n_channels = self.processor_file.n_channels

        self.sai_params = get_sai_params(self.n_channels, self.chunk_size, smoothing_scale=0.5)
        self.sai_file = SAIProcessor(self.sai_params)
        self.vis_file = VisualizationHandler(sample_rate, self.sai_params)

        # File processing
        self.audio_data = None
        self.current_position = 0
        self.duration = 0
        self.total_samples = 0

        # Audio playback
        self.audio_playback_enabled = True
        self.audio_output_stream = None
        self.playback_position = 0.0
        self.playback_finished_flag = False

        self.running = False

        # Practice session
        self.practice_session = None
        self.vocab_list = None
        self.root_path = None

        # Tone selection state (for 2-syllable words, store list of tones)
        self.selected_tones = []  # ordered list of tone selections for current item

        # Results
        self.results = []
        self.script_dir = Path(__file__).parent
        self.save_dir = Path("perception_results")
        self.save_dir.mkdir(exist_ok=True)

        self._setup_visualization()

    def start(self):
        self.running = True
        self.ani = animation.FuncAnimation(
            self.fig, self.update_visualization,
            interval=int((self.chunk_size / self.sample_rate) * 1000),
            blit=False,
            cache_frame_data=False
        )
        print("Starting perception mode visualization...")
        plt.show()

    def stop(self):
        self.running = False
        if self.audio_output_stream:
            try:
                self.audio_output_stream.stop()
                self.audio_output_stream.close()
            except Exception:
                pass
        plt.close(self.fig)
        print("Perception app stopped.")

    # --- CSV SAVING ---
    def _save_results_to_csv(self):
        filename = "session1_perception_results.csv"
        filepath = self.script_dir / filename
        file_exists = filepath.exists()

        try:
            with open(filepath, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=[
                    'item_idx', 'chinese', 'pinyin', 'syllables',
                    'correct_tone', 'user_answer', 'is_correct',
                    'ref_audio', 'timestamp'
                ])
                if not file_exists:
                    writer.writeheader()
                writer.writerows(self.results)
                self.results = []
            print(f"✅ Session log saved to {filepath}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    def _on_playback_finished(self):
        self.playback_finished_flag = True

    def _play_audio_file(self, audio_data, sample_rate):
        if self.audio_output_stream and self.audio_output_stream.active:
            self.audio_output_stream.stop()

        self.playback_position = 0.0

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

    def _load_practice_item(self, item):
        if not item or not self.practice_session:
            return

        # Reset selection state
        self.selected_tones = []

        progress_str = self.practice_session.get_progress_string()
        n_syllables = item.get('syllables', 1)

        # In perception mode, we hide the chinese/pinyin until user answers
        # Show only the prompt
        if n_syllables == 1:
            prompt = f"Identify the tone   ({progress_str})"
        else:
            prompt = f"Identify the tones (in order)   ({progress_str})"

        if hasattr(self, 'practice_text'):
            self.practice_text.set_text(prompt)

        # Update answer display
        self._update_answer_display()

        # Load audio
        audio_path, _ = self.practice_session.get_audio_for_current()

        if audio_path and os.path.exists(audio_path):
            audio_data, original_sr = librosa.load(audio_path, sr=None)
            if original_sr != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=self.sample_rate)
            self.audio_data = audio_data
            self.total_samples = len(self.audio_data)
            self.duration = self.total_samples / self.sample_rate
            self.current_position = 0

            self.vis_file.img[:] = 0
            self.im_file.set_data(self.vis_file.img)
        else:
            print("Audio path invalid or not found.")

        if hasattr(self, 'status_text'):
            self.status_text.set_text('Press Play to listen, then choose tone')
            self.status_text.set_color('yellow')

        self.fig.canvas.draw_idle()

    def _update_answer_display(self):
        """Update the 'Answer' box showing current tone selections"""
        if not hasattr(self, 'answer_text'):
            return

        item = self.practice_session.get_current_item() if self.practice_session else None
        if not item:
            return

        n_syllables = item.get('syllables', 1)

        # Tone shape symbols
        shape_map = {1: '―', 2: '╱', 3: '∨', 4: '╲'}

        if not self.selected_tones:
            if n_syllables == 1:
                display = "Answer:  [ _ ]"
            else:
                display = "Answer:  [ _ ] - [ _ ]"
        else:
            parts = []
            for t in self.selected_tones:
                parts.append(f"[{t} {shape_map.get(t, '?')}]")
            # pad with blanks
            while len(parts) < n_syllables:
                parts.append("[ _ ]")
            display = "Answer:  " + " - ".join(parts)

        self.answer_text.set_text(display)

    def _on_tone_button(self, tone_number):
        """Handle a tone button press."""
        if not self.practice_session:
            return

        item = self.practice_session.get_current_item()
        if not item:
            return

        n_syllables = item.get('syllables', 1)

        # If already complete, ignore (user must press Next or click again to reset)
        if len(self.selected_tones) >= n_syllables:
            # Reset and start a new selection
            self.selected_tones = [tone_number]
        else:
            self.selected_tones.append(tone_number)

        self._update_answer_display()

        # If user has selected all required tones, evaluate
        if len(self.selected_tones) == n_syllables:
            self._evaluate_answer()

        self.fig.canvas.draw_idle()

    def _evaluate_answer(self):
        """Compare user's tone selection to the correct answer"""
        item = self.practice_session.get_current_item()
        if not item:
            return

        # The correct tone string is e.g. "1", "3", "1-2", "4-3"
        correct_tone_str = str(item.get('tone', ''))
        correct_tones = [int(t) for t in correct_tone_str.split('-') if t.strip().isdigit()]

        is_correct = (self.selected_tones == correct_tones)

        if is_correct:
            self.status_text.set_text(f"✓ Correct!  {item['chinese']} ({item['pinyin']})")
            self.status_text.set_color('lime')
        else:
            shape_map = {1: '―', 2: '╱', 3: '∨', 4: '╲'}
            correct_display = " - ".join([f"[{t} {shape_map.get(t, '?')}]" for t in correct_tones])
            self.status_text.set_text(
                f"✗ Not quite. Correct: {correct_display}  →  {item['chinese']} ({item['pinyin']})"
            )
            self.status_text.set_color('#ff6666')

        # Log result
        self.results.append({
            'item_idx': self.practice_session.current_index + 1,
            'chinese': item['chinese'],
            'pinyin': item['pinyin'],
            'syllables': item.get('syllables', 0),
            'correct_tone': correct_tone_str,
            'user_answer': '-'.join(str(t) for t in self.selected_tones),
            'is_correct': is_correct,
            'ref_audio': item.get('audio', 'NA'),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        self._save_results_to_csv()

    def next_item(self, event=None):
        if self.practice_session:
            if self.practice_session.current_index >= self.practice_session.total_items - 1:
                self.status_text.set_text(
                    f"✓ Practice Set Complete ({self.practice_session.total_items}/{self.practice_session.total_items})"
                )
                self.status_text.set_color('lime')
                plt.close(self.fig)
                self._launch_next_script()
            else:
                item = self.practice_session.next_item()
                self._load_practice_item(item)
        self.fig.canvas.draw_idle()

    def toggle_playback(self, event=None):
        if self.audio_output_stream and self.audio_output_stream.active:
            self.audio_output_stream.stop()
            self.btn_playback.label.set_text('▶ Play')
            self.btn_playback.color = 'cyan'
            self.btn_playback.hovercolor = 'lightblue'
        else:
            current_item = self.practice_session.get_current_item()
            if current_item:
                audio_path, _ = self.practice_session.get_audio_for_current()
                if audio_path and os.path.exists(audio_path):
                    audio_data, original_sr = librosa.load(audio_path, sr=None)
                    if original_sr != self.sample_rate:
                        audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=self.sample_rate)
                    self.playback_position = 0.0

                    # Reset pitchogram visualization for fresh display
                    self.vis_file.img[:] = 0
                    self.current_position = 0

                    self._play_audio_file(audio_data, self.sample_rate)
                    self.btn_playback.label.set_text('■ Stop')
                    self.btn_playback.color = '#ff9999'
                    self.btn_playback.hovercolor = '#ff6666'

        self.fig.canvas.draw_idle()

    def get_next_file_chunk(self):
        if self.audio_data is None or self.total_samples == 0:
            return None, -1

        if self.current_position >= self.total_samples:
            return None, -1

        end_position = min(self.current_position + self.chunk_size, self.total_samples)
        chunk = self.audio_data[self.current_position:end_position]

        if len(chunk) < self.chunk_size:
            chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), 'constant')

        chunk_index = self.current_position
        self.current_position = end_position
        return chunk.astype(np.float32), chunk_index

    def update_visualization(self, frame):
        # Reset play button when playback finishes
        if self.playback_finished_flag:
            self.playback_finished_flag = False
            self.btn_playback.label.set_text('▶ Play')
            self.btn_playback.color = 'cyan'
            self.btn_playback.hovercolor = 'lightblue'
            self.fig.canvas.draw_idle()

        try:
            # File SAI Update (the "pitchogram" / reference)
            if self.audio_data is not None:
                # Only advance the pitchogram while audio is actively playing
                is_playing = (self.audio_output_stream is not None
                              and self.audio_output_stream.active)
                if is_playing:
                    chunk, chunk_index = self.get_next_file_chunk()
                    if chunk is not None and chunk_index >= 0:
                        try:
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
                        except Exception:
                            pass

                current_max_file = np.max(self.vis_file.img) if self.vis_file.img.size else 1
                self.im_file.set_data(self.vis_file.img)
                self.im_file.set_clim(vmin=0, vmax=max(1, min(255, current_max_file * 1.3)))
        except Exception:
            pass

        return [self.im_file, self.status_text, self.practice_text, self.answer_text]

    def _launch_next_script(self):
        """Launch the production mode script."""
        target_file = "tone_production_SAI.py"
        current_dir = Path(__file__).parent
        next_script = current_dir / target_file

        if not next_script.exists():
            # Fallback names
            for fallback in ["tone_production_SAI_two_syllable.py", "production_SAI.py"]:
                cand = current_dir / fallback
                if cand.exists():
                    next_script = cand
                    break

        if next_script.exists():
            print(f"🚀 Launching production script: {next_script}")
            subprocess.Popen([sys.executable, str(next_script)])
        else:
            print(f"⚠️ Could not find production script.")

    def _switch_to_production(self):
        """Triggered by pressing '2' — switch to production mode."""
        print("Switching to Production Mode...")
        self.status_text.set_text('Switching to Production mode...')
        self.status_text.set_color('orange')
        self.fig.canvas.draw_idle()
        plt.pause(0.4)
        plt.close(self.fig)
        self._launch_next_script()

    def on_key_press(self, event):
        """Keyboard shortcuts:
        - 1, 2, 3, 4 → tone selection
        - space → play/stop reference
        - n / right → next item
        - 'p' → switch to Production mode
        - 'm' (mode) → switch to Production mode (alternative)
        """
        if event.key in ('1', '2', '3', '4'):
            # Tone selection takes priority for number keys
            self._on_tone_button(int(event.key))
        elif event.key == ' ':
            self.toggle_playback()
        elif event.key in ('n', 'right'):
            self.next_item()
        elif event.key in ('p', 'P'):
            # Switch to production mode
            self._switch_to_production()

    def _setup_visualization(self):
        """Build the perception-mode UI matching the sketch:
        - Pitchogram on top
        - 'Identify the tone(s)' prompt
        - Four tone buttons with shape symbols
        - 'Answer' display
        """
        self.fig = plt.figure(figsize=(13, 9))
        self.fig.canvas.manager.set_window_title("Tone Perception — Press [1] Perception  [2] Production")

        # Layout with GridSpec
        gs = self.fig.add_gridspec(
            4, 1,
            height_ratios=[6, 1.2, 1.5, 1.0],
            hspace=0.35
        )

        # --- 1. Pitchogram (top) ---
        self.ax_file = self.fig.add_subplot(gs[0, 0])
        self.im_file = self.ax_file.imshow(
            self.vis_file.img, aspect='auto', origin='upper',
            interpolation='bilinear', extent=[0, self.sai_width, 0, self.n_channels],
            cmap='jet', vmin=0, vmax=255
        )
        self.ax_file.set_title('Pitchogram', color='cyan', fontsize=28, weight='bold', pad=12)
        self.ax_file.axis('off')

        # Subtle border around pitchogram
        for spine in self.ax_file.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#444')
            spine.set_linewidth(2)

        # --- 2. Prompt text ("Identify the tone(s)") ---
        self.ax_prompt = self.fig.add_subplot(gs[1, 0])
        self.ax_prompt.axis('off')

        self.practice_text = self.ax_prompt.text(
            0.5, 0.55, 'Identify the tone(s)',
            transform=self.ax_prompt.transAxes,
            color='white', ha='center', va='center',
            fontsize=26, weight='bold'
        )

        self.status_text = self.ax_prompt.text(
            0.5, 0.05, 'Press Play to listen, then choose tone',
            transform=self.ax_prompt.transAxes,
            color='yellow', ha='center', va='center',
            fontsize=14
        )

        # --- 3. Tone buttons row ---
        # Four equal-width buttons
        self.ax_tone_row = self.fig.add_subplot(gs[2, 0])
        self.ax_tone_row.axis('off')

        # Tone shape symbols (use unicode geometric chars that render well)
        # 1: high level   ―
        # 2: rising       ╱
        # 3: dipping      ∨
        # 4: falling      ╲
        tone_specs = [
            (1, '―', '#5fbf5f'),  # green
            (2, '╱', '#5fbfbf'),  # teal
            (3, '∨', '#bf8f5f'),  # amber
            (4, '╲', '#bf5f5f'),  # red
        ]

        # Position the four tone buttons evenly across the figure
        n_buttons = 4
        button_width = 0.16
        button_height = 0.11
        total_width = n_buttons * button_width
        spacing = (1.0 - total_width - 0.1) / (n_buttons - 1)  # 0.05 margin each side
        start_x = 0.05

        self.tone_buttons = {}
        self.tone_button_axes = {}

        for i, (tone_num, shape, color) in enumerate(tone_specs):
            x = start_x + i * (button_width + spacing)
            y = 0.18  # vertical position in figure coords
            ax_btn = self.fig.add_axes([x, y, button_width, button_height])
            label = f"{tone_num}  {shape}"
            btn = Button(ax_btn, label, color=color, hovercolor=self._lighten(color))
            btn.label.set_fontsize(28)
            btn.label.set_weight('bold')
            btn.label.set_color('white')
            # Bind handler with correct tone number captured
            btn.on_clicked(lambda event, t=tone_num: self._on_tone_button(t))
            self.tone_buttons[tone_num] = btn
            self.tone_button_axes[tone_num] = ax_btn

        # --- 4. Answer box (bottom) ---
        self.ax_answer = self.fig.add_subplot(gs[3, 0])
        self.ax_answer.axis('off')

        # Draw an outlined "answer box"
        self.answer_text = self.ax_answer.text(
            0.5, 0.5, 'Answer:  [ _ ]',
            transform=self.ax_answer.transAxes,
            color='white', ha='center', va='center',
            fontsize=22, weight='bold',
            bbox=dict(
                boxstyle='round,pad=0.6',
                facecolor='#1e1e1e',
                edgecolor='#888',
                linewidth=2
            )
        )

        # --- Bottom control buttons (Play / Next / Mode) ---
        self.ax_play_button = plt.axes([0.10, 0.04, 0.18, 0.05])
        self.btn_playback = Button(self.ax_play_button, '▶ Play', color='cyan', hovercolor='lightblue')
        self.btn_playback.label.set_fontsize(14)
        self.btn_playback.label.set_weight('bold')
        self.btn_playback.on_clicked(self.toggle_playback)

        self.ax_next_button = plt.axes([0.40, 0.04, 0.18, 0.05])
        self.btn_next = Button(self.ax_next_button, 'Next →', color='orange', hovercolor='gold')
        self.btn_next.label.set_fontsize(14)
        self.btn_next.label.set_weight('bold')
        self.btn_next.on_clicked(self.next_item)

        # Mode toggle button
        self.ax_mode_button = plt.axes([0.70, 0.04, 0.22, 0.05])
        self.btn_mode = Button(
            self.ax_mode_button,
            'Production Mode  [2]',
            color='#444466',
            hovercolor='#6666aa'
        )
        self.btn_mode.label.set_fontsize(12)
        self.btn_mode.label.set_color('white')
        self.btn_mode.label.set_weight('bold')
        self.btn_mode.on_clicked(lambda event: self._switch_to_production())

        # Mode indicator (small, top-right)
        self.fig.text(
            0.99, 0.985, '[1] Perception (active)   [2] Production',
            ha='right', va='top',
            color='#aaaaaa', fontsize=10, style='italic'
        )

        # Background colors
        self.fig.patch.set_facecolor('#121212')
        self.ax_prompt.set_facecolor('#121212')
        self.ax_answer.set_facecolor('#121212')

        plt.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.12)

        # Keyboard shortcuts
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    @staticmethod
    def _lighten(hex_color, amount=0.15):
        """Return a slightly lighter hex color for hover."""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        new_rgb = tuple(min(255, int(c + (255 - c) * amount)) for c in rgb)
        return '#{:02x}{:02x}{:02x}'.format(*new_rgb)


# ---------------- Main Execution ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Tone Perception Mode: identify Mandarin tone(s) from reference audio."
    )
    parser.add_argument("--word", type=str, help="Specify a single Mandarin word for practice.")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    vocab_list = VocabList(root_path=str(script_dir))

    if len(vocab_list.all_items) == 0:
        print("❌ No audio files found in subfolders. Check your directory structure.")
        return 1

    practice_set = None
    word_info = None

    if args.word:
        word_info = next(
            (item for item in vocab_list.all_items
             if item.get('type') == 'word' and item.get('chinese') == args.word),
            None
        )
        if word_info:
            practice_set = {'words': [word_info]}
            print(f"Single word mode: {word_info['chinese']} ({word_info['pinyin']})")
        else:
            print(f"Word '{args.word}' not found.")
            return 1
    else:
        print("--- Starting Perception Mode (3 One-Syl + 3 Two-Syl) ---")
        practice_set = get_random_practice_set_from_vocablist(vocab_list)
        if not practice_set['words']:
            print("❌ No practice items generated.")
            return 1

    app = None
    try:
        app = TonePerceptionApp()
        app.vocab_list = vocab_list
        app.root_path = script_dir

        practice_session = PracticeSession(practice_set, root_path=str(script_dir))
        app.practice_session = practice_session
        app._load_practice_item(practice_session.get_current_item())

        app.start()

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error starting visualization: {e}")
        traceback.print_exc()
        return 1
    finally:
        if app is not None:
            app.stop()
        print("✅ Perception app stopped cleanly")

    return 0


if __name__ == '__main__':
    sys.exit(main())