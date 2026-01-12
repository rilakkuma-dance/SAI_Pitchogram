import sys
import numpy as np
import pyaudio
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
from pathlib import Path

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
    sys.exit(1)

try:
    import torch
    import torchaudio
    WAV2VEC2_AVAILABLE = True
except ImportError:
    WAV2VEC2_AVAILABLE = False

# ---------------- Custom Module Imports ----------------
# Assuming these exist in your 'modules' folder based on your imports
from modules.visualization_handler import VisualizationHandler, SAIParams
from modules.phoneme_handler import PhonemeAnalyzer
from modules.recorder import AudioRecorder
from modules.tone_grader_word import ToneGraderWord, GradingResult

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
    """Get font path relative to script location"""
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

# Initialize Setup
setup_chinese_font()
font_path = get_font_path()
if font_path:
    font_prop = fm.FontProperties(fname=font_path, size=16)
else:
    font_prop = fm.FontProperties(family='Times New Roman', size=16)

# ---------------- Helper Classes ----------------

class VocabList:
    """Manages the hardcoded vocabulary list"""
    def __init__(self, audio_base_path="mandarin_audio_one_syllable"):
        self.all_items = []
        self.audio_base_path = Path(audio_base_path)
        
        # 15 words - WAV format definitions
        words = [
            # Tone 1
            {"id": 1,  "chinese": "天", "pinyin": "tiān", "tone": "1", "audio": "01_天_1.wav", "type": "word"},
            {"id": 2,  "chinese": "心", "pinyin": "xīn",  "tone": "1", "audio": "02_心_1.wav", "type": "word"},
            {"id": 3,  "chinese": "车", "pinyin": "chē",  "tone": "1", "audio": "03_车_1.wav", "type": "word"},
            # Tone 2
            {"id": 4,  "chinese": "学", "pinyin": "xué",  "tone": "2", "audio": "04_学_2.wav", "type": "word"},
            {"id": 5,  "chinese": "人", "pinyin": "rén",  "tone": "2", "audio": "05_人_2.wav", "type": "word"},
            {"id": 6,  "chinese": "白", "pinyin": "bái",  "tone": "2", "audio": "06_白_2.wav", "type": "word"},
            # Tone 3
            {"id": 7,  "chinese": "老", "pinyin": "lǎo",  "tone": "3", "audio": "07_老_3.wav", "type": "word"},
            {"id": 8,  "chinese": "火", "pinyin": "huǒ",  "tone": "3", "audio": "08_火_3.wav", "type": "word"},
            {"id": 9,  "chinese": "狗", "pinyin": "gǒu",  "tone": "3", "audio": "09_狗_3.wav", "type": "word"},
            # Tone 4
            {"id": 10, "chinese": "叫", "pinyin": "jiào", "tone": "4", "audio": "10_叫_4.wav", "type": "word"},
            {"id": 11, "chinese": "骂", "pinyin": "mà",   "tone": "4", "audio": "11_骂_4.wav", "type": "word"},
            {"id": 12, "chinese": "去", "pinyin": "qù",   "tone": "4", "audio": "12_去_4.wav", "type": "word"},
        ]
        
        all_potential_items = words
        
        # Filter to only include items whose audio files actually exist
        self.all_items = []
        missing_files = []
        for item in all_potential_items:
            audio_path = self.audio_base_path / item['audio']
            if audio_path.exists():
                self.all_items.append(item)
            else:
                missing_files.append(str(audio_path))
        
        print(f"\nWAV audio files found: {len(self.all_items)} / {len(all_potential_items)}")
        if missing_files and len(missing_files) <= 10:
            print(f"Missing WAV files ({len(missing_files)}):")
            for f in missing_files:
                print(f"  - {f}")

class PracticeSession:
    """Manages practice session with multiple words and sentences"""
    def __init__(self, practice_set, audio_manager, audio_base_path='mandarin_audio_one_syllable'):
        self.practice_set = practice_set
        self.audio_manager = audio_manager
        self.audio_base_path = Path(audio_base_path)
        self.practice_session = None
        self.current_index = 0
        self.all_items = practice_set['words'] + practice_set['sentences'] # Combine them
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
        
        # Using the direct filename from VocabList
        audio_filename = item.get('audio')
        if not audio_filename: return None, None

        audio_path = self.audio_base_path / audio_filename
        
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
    """Minimal Manager to satisfy PracticeSession dependency"""
    def __init__(self, base_dir='audio'):
        self.base_dir = base_dir

def get_random_practice_set_from_vocablist(vocab_list):
    """Get random practice items from VocabList instance"""
    import random
    practice_set = {'words': [], 'sentences': []}
    
    # Separate words and sentences
    words = [item for item in vocab_list.all_items if item.get('type') == 'word']
    sentences = [item for item in vocab_list.all_items if item.get('type') == 'sentence']
    
    # Get random words (up to 5)
    if len(words) >= 3:
        practice_set['words'] = random.sample(words, 3)
    else:
        practice_set['words'] = words
    
    return practice_set

# ---------------- Processing Handlers ----------------

class SimpleWav2Vec2Handler:
    """Wav2Vec2 phoneme recognition handler"""
    def __init__(self, model_name="facebook/wav2vec2-xlsr-53-espeak-cv-ft", sample_rate=16000, target_phonemes="ɕiɛɕiɛ"):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.enabled = WAV2VEC2_AVAILABLE
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None
        
        # Audio buffer for mic recording
        self.audio_buffer = []
        self.is_recording = False
        self.is_processing = False
        self.result = None
        self.target_phonemes = target_phonemes
        self.phoneme_analyzer = PhonemeAnalyzer(self.target_phonemes)
        self.analysis_results = None
        self.overall_score = 0.0
        self.callbacks = []

        if self.enabled:
            # Placeholder for actual model loading logic if needed
            print("Wav2Vec2 enabled (Mock mode for structure compatibility)")

    def register_callback(self, callback, *args):
        self.callbacks.append((callback, args))

    def run_callbacks(self, complete_audio):
        for callback, args in self.callbacks:
            try:
                callback(complete_audio, self.result, self.overall_score, *args)
            except Exception as e:
                print(f"Callback error: {e}")

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
                 debug=True, playback_speed=1.0, loop_audio=True):

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

        # SAI parameters
        self.sai_params = SAIParams(
            num_channels=self.n_channels,
            sai_width=400,
            future_lags=399,
            num_triggers_per_frame=2,
            trigger_window_width=self.chunk_size + 1,
            input_segment_width=self.chunk_size,
            channel_smoothing_scale=0.5
        )
        
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
        
        # --- PRACTICE MODE INTEGRATION ---
        self.voice_selector = VoiceSelector()
        self.audio_manager = AudioManager()
        self.practice_session = None 
        self.wav2vec2_handler = SimpleWav2Vec2Handler(sample_rate=sample_rate, target_phonemes=self.target_phonemes)
        self.wav2vec2_handler.register_callback(self._handle_processing_complete)
        
        # Simple local audio recorder
        self.is_recording_simple = False
        self.recorder = AudioRecorder(sample_rate=self.sample_rate)
        self.vocab_list = None
        self.audio_base_path = None
        
        # Tone grader
        self.grader = ToneGraderWord()
        self.recorder.add_audio_callback(self._grade_recording)

        self._setup_dual_visualization()

    def _play_audio_file(self, audio_data, sample_rate):
        if self.audio_output_stream and self.audio_output_stream.active:
            self.audio_output_stream.stop()
        
        if self.audio_data is not None and self.audio_output_stream:
            self.audio_output_stream.close()

        if audio_data is None or audio_data.size == 0:
            print("No audio data to play.")
            self.audio_data = None
            self.total_samples = 0
            return

        self.audio_data = audio_data.copy()
        if np.max(np.abs(self.audio_data)) > 0:
            self.audio_data = self.audio_data / np.max(np.abs(self.audio_data)) * 0.9
        
        self.total_samples = len(self.audio_data)
        self.duration = self.total_samples / sample_rate
        self.current_position = 0
        self.playback_position = 0.0
        self.loop_count = 0
        
        try:
            self.audio_output_stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=self._audio_playback_callback
            )
            self.audio_output_stream.start()
            print(f"Playing reference audio ({self.duration:.1f}s)")
        except Exception as e:
            print(f"Failed to create audio playback: {e}")
            self.audio_playback_enabled = False

    def _get_item_display(self, item):
        if not item: return "End of Set"
        if 'chinese' in item and item.get('type') == 'word':
            return f"WORD: {item['chinese']} ({item['pinyin']}) - {item['english'] if 'english' in item else ''}"
        else:
            return f"SENTENCE: {item['chinese']}"

    def _load_practice_item(self, item):
        if not item or not self.practice_session: return
        
        # 1. Update internal phoneme/audio data
        reference_pronunciation = item.get('pinyin', item.get('chinese'))
        translation = item.get('english', '')
        target_phonemes = item.get('phonemes', 'placeholder')
        
        self.set_reference_text(target_phonemes, reference_pronunciation, translation)
        self.wav2vec2_handler.target_phonemes = target_phonemes
        self.clear_phoneme_feedback()

        # 2. Update the UI Text (Centered, White, Bold)
        # Format: 中国 (zhōngguó) - 1/10
        progress_str = self.practice_session.get_progress_string()
        display_text = f"{item['chinese']} ({item['pinyin']}) - {progress_str}"
        
        if hasattr(self, 'practice_text'):
            self.practice_text.set_text(display_text)

        # 3. Handle Audio Loading
        audio_path, _ = self.practice_session.get_audio_for_current(self.voice_selector.current_voice)
        if audio_path and os.path.exists(audio_path):
            audio_data, original_sr = librosa.load(audio_path, sr=None)
            if original_sr != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=self.sample_rate)
            self.audio_data = audio_data
            self.total_samples = len(self.audio_data)
            self.current_position = 0
            
            # Clear reference view until played
            self.vis_file.img[:] = 0
            self.im_file.set_data(self.vis_file.img)
        
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

    def _handle_processing_complete(self, user_audio, transcription, score):
        print(f"Processing complete: Transcription='{transcription}', Score={score:.2f}")
        if hasattr(self, 'status_text'):
            if transcription and transcription != "no_audio":
                self.status_text.set_text(f'Score: {score:.1f}% | {transcription[:30]}...')
                self.status_text.set_color('yellow' if score < 70 else 'lime')
            else:
                self.status_text.set_text('No audio detected')
                self.status_text.set_color('orange')

        self.vis_realtime.img[:] = 0
        self.im_realtime.set_data(self.vis_realtime.img)
        
        def _process_user_sai():
            processor = AudioProcessor(self.sample_rate)
            sai_processor = SAIProcessor(self.sai_params)
            total_frames = len(user_audio) // self.chunk_size
            
            for i in range(total_frames):
                chunk = user_audio[i * self.chunk_size : (i + 1) * self.chunk_size]
                nap_output = processor.process_chunk(chunk)
                sai_output = sai_processor.RunSegment(nap_output)
                self.vis_realtime.get_vowel_embedding(nap_output)
                self.vis_realtime.run_frame(sai_output)
                
                if self.vis_realtime.img.shape[1] > 1:
                    self.vis_realtime.img[:, :-1] = self.vis_realtime.img[:, 1:]
                    self.vis_realtime.draw_column(self.vis_realtime.img[:, -1])
            self.fig.canvas.draw_idle()

        threading.Thread(target=_process_user_sai, daemon=True).start()

    def decrease_sai_speed(self, event=None):
        self.sai_speed = max(0.1, self.sai_speed - 0.25)
    def increase_sai_speed(self, event=None):
        self.sai_speed = min(5.0, self.sai_speed + 0.25)
    def decrease_audio_speed(self, event=None):
        self.playback_speed = max(0.25, self.playback_speed - 0.25)
    def increase_audio_speed(self, event=None):
        self.playback_speed = min(5.0, self.playback_speed + 0.25)

    def toggle_voice(self, event=None):
        new_voice = self.voice_selector.toggle()
        # Note: self.btn_voice is not defined in setup_dual_visualization in original code, 
        # but logic exists. Assuming logic handles re-load.
        self._load_practice_item(self.practice_session.get_current_item())

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
                
                if self.loop_audio and np.any(chunk_indices >= self.total_samples):
                    chunk_indices = chunk_indices % self.total_samples
                    
                chunk_indices = np.clip(chunk_indices, 0, self.total_samples - 1)
                chunk = self.audio_data[chunk_indices]

                outdata[:len(chunk), 0] = chunk
                outdata[len(chunk):, 0].fill(0)
                
                self.playback_position += int(frames * speed_factor)
                if self.playback_position >= self.total_samples and self.total_samples > 0:
                    if self.loop_audio:
                        self.playback_position = self.playback_position % self.total_samples
                    else:
                        outdata.fill(0)
                        raise sd.CallbackStop
            else:
                outdata.fill(0)
        except sd.CallbackStop:
            raise
        except Exception:
            outdata.fill(0)

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
        if self.p is None:
            self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_input_callback
        )

    def _audio_input_callback(self, in_data, frame_count, time_info, status):
        try:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            if not self.audio_queue.full():
                self.audio_queue.put(audio_data)
        except Exception:
            pass
        return (in_data, pyaudio.paContinue)

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
        try:
            # File SAI Update
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

            # User SAI Update
            current_max_rt = np.max(self.vis_realtime.img) if self.vis_realtime.img.size > 0 else 1
            self.im_realtime.set_data(self.vis_realtime.img)
            self.im_realtime.set_clim(vmin=0, vmax=max(1, min(255, current_max_rt * 1.3)))

        except Exception:
            pass
        return [self.im_realtime, self.im_file, self.status_text, self.practice_text, self.progress_text]

    def _save_recording_with_metadata(self, audio_data):
        try:
            save_dir = Path("recordings")
            save_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            wav_filename = f"sai_{timestamp}.wav"
            txt_filename = f"sai_{timestamp}.txt"
            wav_path = save_dir / wav_filename
            txt_path = save_dir / txt_filename
            
            if np.max(np.abs(audio_data)) > 0:
                audio_normalized = audio_data / np.max(np.abs(audio_data)) * 0.95
            else:
                audio_normalized = audio_data
            
            audio_int16 = (audio_normalized * 32767).astype(np.int16)
            
            with wave.open(str(wav_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            current_item = self.practice_session.get_current_item()
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Recording: {wav_filename}\n")
                f.write(f"Item: {current_item.get('chinese', 'N/A')}\n")
            
            print(f"Recording saved: {wav_path}")
            if hasattr(self, 'status_text'):
                self.status_text.set_text(f'Saved: {wav_filename}')
                self.status_text.set_color('lime')
                self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"❌ Error saving recording: {e}")

    def _grade_recording(self, audio_data):
        if not self.practice_session: return
        current_item = self.practice_session.get_current_item()
        chinese = current_item.get('chinese', None)
        pinyin = current_item.get('pinyin', None)
        english = current_item.get('english', None)
        tones = current_item.get('tone', None)

        if type(tones) == list: tones = tones[0]
        self.grader.grade_audio(audio_data, chinese, pinyin, english, tones)
        
        save_dir = Path("recordings")
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        txt_filename = f"sai_{timestamp}_grade.txt"
        self.grader.save_results(save_dir / txt_filename)

    def start(self):
        self.running = True
        self._setup_audio_playback()
        self._setup_mic_stream()
        threading.Thread(target=self.process_realtime_audio, daemon=True).start()
        self.ani = animation.FuncAnimation(
            self.fig, self.update_visualization, 
            interval=int((self.chunk_size / self.sample_rate) * 1000), blit=True
        )
        print("Starting visualization...")
        plt.show()

    def stop(self):
        self.running = False
        if self.audio_output_stream:
            self.audio_output_stream.stop()
            self.audio_output_stream.close()
        sd.stop()
        plt.close(self.fig)
        print("SAIVisualizationWithWav2Vec2 stopped.")

    def _setup_dual_visualization(self):
        self.fig = plt.figure(figsize=(14, 8))
        # Matches the Spectrogram grid: High visual area, medium text area, small control area
        gs = self.fig.add_gridspec(3, 2, height_ratios=[6, 1.5, 0.5])

        # LEFT SAI display (Your Audio - Live)
        self.ax_realtime = self.fig.add_subplot(gs[0, 0])
        self.im_realtime = self.ax_realtime.imshow(
            self.vis_realtime.img, aspect='auto', origin='upper',
            interpolation='bilinear', extent=[0, self.sai_width, 0, self.n_channels],
            cmap='magma', vmin=0, vmax=255
        )
        self.ax_realtime.set_title('Your Audio (Live)', color='lime', fontsize=12, weight='bold')
        self.ax_realtime.axis('off')

        # RIGHT SAI display (Reference Audio)
        self.ax_file = self.fig.add_subplot(gs[0, 1])
        self.im_file = self.ax_file.imshow(
            self.vis_file.img, aspect='auto', origin='upper',
            interpolation='bilinear', extent=[0, self.sai_width, 0, self.n_channels],
            cmap='magma', vmin=0, vmax=255
        )
        self.ax_file.set_title('Reference Pattern', color='cyan', fontsize=12, weight='bold')
        self.ax_file.axis('off')

        # Practice info area (Lower Middle)
        self.ax_practice = self.fig.add_subplot(gs[1, :])
        self.ax_practice.axis('off')
        
        current_item = self.practice_session.get_current_item() if self.practice_session else None
        item_text = ""
        if current_item:
            progress = self.practice_session.get_progress_string()
            item_text = f"{current_item['chinese']} ({current_item['pinyin']}) - {progress}"
        
        # Center-aligned bold text exactly like the spectrogram UI
        self.practice_text = self.ax_practice.text(
            0.5, 0.6, item_text, transform=self.ax_practice.transAxes,
            color='white', ha='center', fontsize=16, weight='bold'
        )

        self.status_text = self.ax_practice.text(
            0.5, 0.2, 'Ready', transform=self.ax_practice.transAxes,
            color='yellow', ha='center', fontsize=11
        )

        # Progress tracking (optional hidden text for logic compatibility)
        self.progress_text = self.ax_practice.text(0, 0, "", alpha=0) 

        # --- Buttons (Matching the Spectrogram UI positions/colors) ---
        from matplotlib.widgets import Button
        
        # 1. Play Reference Button
        self.ax_play_button = plt.axes([0.25, 0.05, 0.15, 0.04])
        self.btn_playback = Button(self.ax_play_button, 'Play Reference', color='cyan', hovercolor='lightblue')
        self.btn_playback.on_clicked(self.toggle_playback)

        # 2. Record / Stop & Save Button
        self.ax_rec_button = plt.axes([0.42, 0.05, 0.18, 0.04])
        self.btn_record = Button(self.ax_rec_button, 'Start Recording', color='lime', hovercolor='green')
        self.btn_record.on_clicked(self.toggle_record)

        # 3. Next Item Button
        self.ax_next_button = plt.axes([0.62, 0.05, 0.15, 0.04])
        self.btn_next = Button(self.ax_next_button, 'Next Item', color='orange', hovercolor='yellow')
        self.btn_next.on_clicked(self.next_item)

        # Background color
        self.fig.patch.set_facecolor('#121212')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.1, hspace=0.2)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def toggle_record(self, event=None):
        if self.wav2vec2_handler.is_processing:
            print("⚠️ Processing... please wait.")
            return

        if not self.is_recording_simple:
            # START RECORDING
            self.is_recording_simple = True
            self.btn_record.label.set_text('Stop & Save')
            self.btn_record.ax.set_facecolor('#ff4444') # Brighter red
            self.recorder.start_recording()
            self.clear_phoneme_feedback()
            
            if hasattr(self, 'status_text'):
                self.status_text.set_text('● Recording... Speak now!')
                self.status_text.set_color('red')
            self.fig.canvas.draw_idle()
        else:
            # STOP AND AUTOMATICALLY SAVE
            self.is_recording_simple = False
            self.btn_record.label.set_text('Start Record')
            self.btn_record.ax.set_facecolor('lightgreen')
            
            # This triggers the callback that handles grading/processing
            recorded_audio = self.recorder.stop_recording() 
            
            # Automatically trigger save with metadata
            if recorded_audio is not None:
                self._save_recording_with_metadata(recorded_audio)
                
            if hasattr(self, 'status_text'):
                self.status_text.set_text('Processing & Saved!')
                self.status_text.set_color('orange')
            self.fig.canvas.draw_idle()

    def next_item(self, event=None):
        if self.practice_session:
            # Check if we are about to wrap around
            if self.practice_session.current_index >= self.practice_session.total_items - 1:
                self.status_text.set_text(f"✓ Practice Set Complete ({self.practice_session.total_items}/{self.practice_session.total_items})")
                self.status_text.set_color('lime')
                # Optional: Uncomment the next line to loop back to the start
                # item = self.practice_session.next_item()
                # self._load_practice_item(item)
            else:
                item = self.practice_session.next_item()
                self._load_practice_item(item)
        self.fig.canvas.draw_idle()

    def toggle_playback(self, event=None):
        if self.audio_output_stream and self.audio_output_stream.active:
            self.audio_output_stream.stop()
            self.btn_playback.label.set_text('Play')
        else:
            current_item = self.practice_session.get_current_item()
            if current_item:
                audio_path, _ = self.practice_session.get_audio_for_current(self.voice_selector.current_voice)
                if audio_path and os.path.exists(audio_path):
                    audio_data, original_sr = librosa.load(audio_path, sr=None)
                    if original_sr != self.sample_rate:
                        audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=self.sample_rate)
                    self._play_audio_file(audio_data, self.sample_rate)
                    self.btn_playback.label.set_text('Stop Ref')
                else:
                    print(f"⚠️ Cannot play audio: File not found at {audio_path}")

# ---------------- Main Execution ----------------

def main():
    parser = argparse.ArgumentParser(description="SAI Visualization and Mandarin Pronunciation Practice Tool.")
    parser.add_argument("--word", type=str, help="Specify a single Mandarin word for practice.")
    parser.add_argument("--sentence", type=int, help="Specify a sentence ID for practice.")
    args = parser.parse_args()

    # Setup audio base path
    script_dir = Path(__file__).parent
    audio_base = script_dir / 'mandarin_audio_one_syllable'
    
    # Initialize VocabList
    vocab_list = VocabList(audio_base_path=str(audio_base))
    
    if len(vocab_list.all_items) == 0:
        print("❌ No audio files found. Check your reference directory structure.")
        # Proceeding anyway to show UI but warn
    
    practice_set = None
    audio_file_path = None
    word_info = None

    if args.word:
        word_info = next((item for item in vocab_list.all_items 
                          if item.get('type') == 'word' and item.get('chinese') == args.word), None)
        if word_info:
            audio_file_path = str(audio_base / word_info.get('audio'))
            print(f"Single word mode: {word_info['chinese']} ({word_info['pinyin']})")
        else:
            print(f"Word '{args.word}' not found in vocabulary.")
            return 1

    elif args.sentence:
        # Note: Vocabulary list in this version only has words, but logic remains for future extension
        print("Sentence mode not fully supported with current hardcoded vocab list.")
        return 1
    
    else:
        # Practice Mode
        print("--- Starting in Practice Mode (5 random items) ---")
        practice_set = get_random_practice_set_from_vocablist(vocab_list)
        if not practice_set['words'] and not practice_set['sentences']:
            print("❌ No practice items found.")
            return 1
        word_info = practice_set['words'][0] if practice_set['words'] else practice_set['sentences'][0]

    try:
        sai_vis = SAIVisualizationWithWav2Vec2(
            audio_file_path=audio_file_path,
            playback_speed=1.0, 
            loop_audio=(practice_set is not None)
        )

        sai_vis.vocab_list = vocab_list
        sai_vis.audio_base_path = audio_base
        
        if practice_set:
            practice_session = PracticeSession(practice_set, sai_vis.audio_manager, audio_base_path=str(audio_base))
            sai_vis.practice_session = practice_session
            sai_vis._load_practice_item(practice_session.get_current_item())
            
        elif word_info:
            reference_pronunciation = word_info.get('pinyin', word_info.get('chinese'))
            translation = word_info.get('english', '')
            target_phonemes = word_info.get('phonemes', 'placeholder')
            sai_vis.set_reference_text(target_phonemes, reference_pronunciation, translation)
            sai_vis.wav2vec2_handler.target_phonemes = target_phonemes
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