import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.font_manager as fm
from matplotlib.widgets import Button
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
    print("Warning: No Chinese font found. Chinese characters may not display correctly.")
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

        # --- 2. Define Two-Syllable Words ---
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
        if not item: return None, None
        
        audio_filename = item.get('audio')
        folder_name = item.get('folder', 'mandarin_audio_one_syllable')
        if not audio_filename: return None, None
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
    
    print(f"Generated Practice Set: {len(combined_words)} items (Target: 6)")
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

class SAIVisualizationWithWav2Vec2:
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
        
        # --- NEW ATTRIBUTES FOR SAVING ---
        self.results = []
        self.script_dir = Path(__file__).parent
        self.save_dir = Path("recordings")
        self.save_dir.mkdir(exist_ok=True)
        # ---------------------------------

        print("⚠️ Forcing Dummy Recorder for Lab Computer.")
        self.recorder = DummyRecorder(sample_rate=self.sample_rate)
        
        self._setup_dual_visualization()

    def start(self):
        self.running = True
        self._setup_audio_playback()
        self._setup_mic_stream()
        threading.Thread(target=self.process_realtime_audio, daemon=True).start()
        self.ani = animation.FuncAnimation(
            self.fig, self.update_visualization, 
            interval=int((self.chunk_size / self.sample_rate) * 1000), 
            blit=True,
            cache_frame_data=False
        )
        print("Starting visualization...")
        plt.show()

    def stop(self):
        self.running = False
        if self.audio_output_stream:
            self.audio_output_stream.stop()
            self.audio_output_stream.close()
        if hasattr(self, 'input_stream') and self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
        plt.close(self.fig)
        print("SAIVisualization stopped.")

    # --- CSV SAVING METHOD ---
    def _save_results_to_csv(self):
        filename = "session2_SAI_results.csv" 
        filepath = self.script_dir / filename
        file_exists = filepath.exists()
        
        try:
            with open(filepath, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=[
                    'item_idx', 'chinese', 'pinyin', 'syllables', 
                    'ref_audio', 'user_recording', 'timestamp'
                ])
                if not file_exists:
                    writer.writeheader()
                writer.writerows(self.results)
                self.results = [] 
            print(f"✅ Session log saved to {filepath}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    # --- AUDIO SAVING METHOD ---
    def save_recorded_audio(self, audio_data):
        if audio_data is None or len(audio_data) == 0:
            self.status_text.set_text("No audio captured.")
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CORRECTED: use self.practice_session instead of self.practice_set
        item = self.practice_session.get_current_item()
        
        if not item:
            item = {'chinese': 'Unknown', 'pinyin': 'unknown', 'syllables': 0, 'audio': 'unknown.wav'}
            
        filename = f"rec_{item['chinese']}_{timestamp}.wav"
        path = self.save_dir / filename
        
        try:
            if np.max(np.abs(audio_data)) > 0:
                audio_normalized = audio_data / np.max(np.abs(audio_data)) * 0.95
            else:
                audio_normalized = audio_data
            audio_int16 = (audio_normalized * 32767).astype(np.int16)

            with wave.open(str(path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            # Log Result
            self.results.append({
                # CORRECTED: use self.practice_session
                'item_idx': self.practice_session.current_index + 1,
                'chinese': item['chinese'],
                'pinyin': item['pinyin'],
                'syllables': item.get('syllables', 0),
                'ref_audio': item.get('audio', 'NA'), 
                'user_recording': filename,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            self._save_results_to_csv()
            
            self.status_text.set_text(f"✓ Saved: {filename}")
            self.status_text.set_color('lime')
        except Exception as e:
            print(f"Save error: {e}")
            self.status_text.set_text(f"Error: {str(e)}")

    def _on_playback_finished(self):
        """Called by sounddevice when stream stops (naturally or manually)"""
        self.playback_finished_flag = True

    def _play_audio_file(self, audio_data, sample_rate):
        # ... existing cleanup code ...
        if self.audio_output_stream and self.audio_output_stream.active:
            self.audio_output_stream.stop()
        
        # ... existing setup code ...
        self.playback_position = 0.0
        self.loop_count = 0
        
        try:
            self.audio_output_stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=self._audio_playback_callback,
                finished_callback=self._on_playback_finished  # <--- FIX: Add this line
            )
            self.audio_output_stream.start()
            print(f"Playing reference audio ({self.duration:.1f}s)")
        except Exception as e:
            print(f"Failed to create audio playback: {e}")
            self.audio_playback_enabled = False

    def _load_practice_item(self, item):
        if not item or not self.practice_session: return
        
        reference_pronunciation = item.get('pinyin', item.get('chinese'))
        translation = item.get('english', '')
        target_phonemes = item.get('phonemes', 'placeholder')
        
        self.set_reference_text(target_phonemes, reference_pronunciation, translation)
        self.clear_phoneme_feedback()

        progress_str = self.practice_session.get_progress_string()
        display_text = f"{item['chinese']} ({item['pinyin']}) - {progress_str}"
        
        if hasattr(self, 'practice_text'):
            self.practice_text.set_text(display_text)

        audio_path, _ = self.practice_session.get_audio_for_current(self.voice_selector.current_voice)
        
        if audio_path and os.path.exists(audio_path):
            audio_data, original_sr = librosa.load(audio_path, sr=None)
            if original_sr != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=self.sample_rate)
            self.audio_data = audio_data
            self.total_samples = len(self.audio_data)
            self.current_position = 0
            
            self.vis_file.img[:] = 0
            self.im_file.set_data(self.vis_file.img)
        else:
            print("Audio path invalid or not found.")
        
        if hasattr(self, 'status_text'):
            self.status_text.set_text('Ready')
            self.status_text.set_color('yellow')
            
        self.fig.canvas.draw_idle()

    def clear_phoneme_feedback(self, event=None):
        self.vis_realtime.img[:] = 0
        self.im_realtime.set_data(self.vis_realtime.img)
        if hasattr(self, 'status_text'):
            self.status_text.set_text('Ready')
            self.status_text.set_color('lime')
        self.fig.canvas.draw_idle()

    def decrease_sai_speed(self, event=None):
        self.sai_speed = max(0.1, self.sai_speed - 0.25)
    def increase_sai_speed(self, event=None):
        self.sai_speed = min(5.0, self.sai_speed + 0.25)
    def decrease_audio_speed(self, event=None):
        self.playback_speed = max(0.25, self.playback_speed - 0.25)
    def increase_audio_speed(self, event=None):
        self.playback_speed = min(5.0, self.playback_speed + 0.25)

    def on_key_press(self, event):
        if event.key == 'up' or event.key == '+': self.increase_sai_speed()
        elif event.key == 'down' or event.key == '-': self.decrease_sai_speed()
        elif event.key == 'right': self.increase_audio_speed()
        elif event.key == 'left': self.decrease_audio_speed()
        elif event.key == 'r':
            self.sai_speed = 1.0
            self.playback_speed = 1.0
        elif event.key == 'c': self.clear_phoneme_feedback()

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
                audio_chunk = self.audio_queue.get(timeout=0.1)
                nap_output = self.processor_realtime.process_chunk(audio_chunk)
                sai_output = self.sai_realtime.RunSegment(nap_output)
                self.vis_realtime.get_vowel_embedding(nap_output)
                self.vis_realtime.run_frame(sai_output)
                
                if self.vis_realtime.img.shape[1] > 1:
                    self.vis_realtime.img[:, :-1] = self.vis_realtime.img[:, 1:]
                    self.vis_realtime.draw_column(self.vis_realtime.img[:, -1])
            except queue.Empty:
                continue

    def update_visualization(self, frame):
        # --- 1. Check if audio finished naturally or was stopped ---
        if self.playback_finished_flag:
            self.playback_finished_flag = False
            # Reset Button UI to "Ready" state
            self.btn_playback.label.set_text('Play Reference')
            self.btn_playback.color = 'cyan'
            self.btn_playback.hovercolor = 'lightblue'
            # Force a canvas redraw to update the button immediately
            self.fig.canvas.draw_idle()

        try:
            # --- 2. File SAI Update (Reference) ---
            if self.audio_data is not None:
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

            # --- 3. User SAI Update (Realtime) ---
            # The heavy lifting is done in process_realtime_audio thread; 
            # we just grab the latest image state here.
            current_max_rt = np.max(self.vis_realtime.img) if self.vis_realtime.img.size > 0 else 1
            self.im_realtime.set_data(self.vis_realtime.img)
            self.im_realtime.set_clim(vmin=0, vmax=max(1, min(255, current_max_rt * 1.3)))

        except Exception:
            pass
            
        return [self.im_realtime, self.im_file, self.status_text, self.practice_text, self.progress_text]
    
    def toggle_record(self, event=None):
        try:
            if not self.is_recording_simple:
                print("Attempting to start recording...")
                self.is_recording_simple = True
                
                self.btn_record.label.set_text('Stop & Save')
                self.btn_record.ax.set_facecolor('#ff4444')
                
                self.recorder.start_recording()
                self.clear_phoneme_feedback()
                
                if hasattr(self, 'status_text'):
                    self.status_text.set_text('● Recording... Speak now!')
                    self.status_text.set_color('red')
                self.fig.canvas.draw_idle()
            else:
                self.is_recording_simple = False
                
                self.btn_record.label.set_text('Start Record')
                self.btn_record.ax.set_facecolor('lightgreen')
                
                recorded_audio = self.recorder.stop_recording() 
                
                if recorded_audio is not None:
                    self.save_recorded_audio(recorded_audio)
                else:
                    print("Warning: No audio data received from recorder.")
                
                self.fig.canvas.draw_idle()

        except Exception as e:
            print(f"\n⚠️ RECORDING ERROR: {e}")
            self.is_recording_simple = False
            self.btn_record.label.set_text('Start Record')
            self.btn_record.ax.set_facecolor('gray')
            if hasattr(self, 'status_text'):
                self.status_text.set_text('Microphone Error')
                self.status_text.set_color('red')
            self.fig.canvas.draw_idle()

    def next_item(self, event=None):
        if self.practice_session:
            if self.practice_session.current_index >= self.practice_session.total_items - 1:
                self.status_text.set_text(f"✓ Practice Set Complete ({self.practice_session.total_items}/{self.practice_session.total_items})")
                self.status_text.set_color('lime')
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
            self.btn_playback.label.set_text('Play Reference')
            self.btn_playback.color = 'cyan'          # Reset to default color
            self.btn_playback.hovercolor = 'lightblue'
        else:
            # --- STARTING PLAYBACK ---
            current_item = self.practice_session.get_current_item()
            if current_item:
                audio_path, _ = self.practice_session.get_audio_for_current(self.voice_selector.current_voice)
                if audio_path and os.path.exists(audio_path):
                    audio_data, original_sr = librosa.load(audio_path, sr=None)
                    if original_sr != self.sample_rate:
                        audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=self.sample_rate)
                    
                    self.playback_position = 0.0 
                    self._play_audio_file(audio_data, self.sample_rate)
                    
                    self.btn_playback.label.set_text('Stop Ref')
                    self.btn_playback.color = '#ff9999'       # Set to Light Red to indicate activity
                    self.btn_playback.hovercolor = '#ff6666'  # Darker Red for hover
        
        # Force the UI to redraw immediately so the label/color update shows up
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

    def _setup_dual_visualization(self):
        self.fig = plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(3, 2, height_ratios=[6, 1.5, 0.5])

        self.ax_realtime = self.fig.add_subplot(gs[0, 0])
        self.im_realtime = self.ax_realtime.imshow(
            self.vis_realtime.img, aspect='auto', origin='upper',
            interpolation='bilinear', extent=[0, self.sai_width, 0, self.n_channels],
            cmap='jet', vmin=0, vmax=255  # <--- CHANGED TO JET FOR HIGH CONTRAST
        )
        self.ax_realtime.set_title('Your Audio (Live)', color='lime', fontsize=12, weight='bold')
        self.ax_realtime.axis('off')

        self.ax_file = self.fig.add_subplot(gs[0, 1])
        self.im_file = self.ax_file.imshow(
            self.vis_file.img, aspect='auto', origin='upper',
            interpolation='bilinear', extent=[0, self.sai_width, 0, self.n_channels],
            cmap='jet', vmin=0, vmax=255  # <--- CHANGED TO JET FOR HIGH CONTRAST
        )
        self.ax_file.set_title('Reference Pattern', color='cyan', fontsize=12, weight='bold')
        self.ax_file.axis('off')

        self.ax_practice = self.fig.add_subplot(gs[1, :])
        self.ax_practice.axis('off')
        
        current_item = self.practice_session.get_current_item() if self.practice_session else None
        item_text = ""
        if current_item:
            progress = self.practice_session.get_progress_string()
            item_text = f"{current_item['chinese']} ({current_item['pinyin']}) - {progress}"
        
        self.practice_text = self.ax_practice.text(
            0.5, 0.6, item_text, transform=self.ax_practice.transAxes,
            color='white', ha='center', fontsize=16, weight='bold'
        )

        self.status_text = self.ax_practice.text(
            0.5, 0.2, 'Ready', transform=self.ax_practice.transAxes,
            color='yellow', ha='center', fontsize=11
        )

        self.progress_text = self.ax_practice.text(0, 0, "", alpha=0) 

        from matplotlib.widgets import Button
        
        self.ax_play_button = plt.axes([0.25, 0.05, 0.15, 0.04])
        self.btn_playback = Button(self.ax_play_button, 'Play Reference', color='cyan', hovercolor='lightblue')
        self.btn_playback.on_clicked(self.toggle_playback)

        self.ax_rec_button = plt.axes([0.42, 0.05, 0.18, 0.04])
        self.btn_record = Button(self.ax_rec_button, 'Start Recording', color='lime', hovercolor='green')
        self.btn_record.on_clicked(self.toggle_record)

        self.ax_next_button = plt.axes([0.62, 0.05, 0.15, 0.04])
        self.btn_next = Button(self.ax_next_button, 'Next Item', color='orange', hovercolor='yellow')
        self.btn_next.on_clicked(self.next_item)

        self.fig.patch.set_facecolor('#121212')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.1, hspace=0.2)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
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
        word_info = next((item for item in vocab_list.all_items 
                          if item.get('type') == 'word' and item.get('chinese') == args.word), None)
        if word_info:
            audio_file_path = str(script_dir / word_info['folder'] / word_info['audio'])
            print(f"Single word mode: {word_info['chinese']} ({word_info['pinyin']})")
        else:
            print(f"Word '{args.word}' not found in vocabulary.")
            return 1
    
    else:
        print("--- Starting in Practice Mode (3 One-Syl + 3 Two-Syl) ---")
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
    sys.exit(main())