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

setup_chinese_font()


# ---------------- Visualization Handler ----------------
from modules.visualization_handler import VisualizationHandler, SAIParams

# Phoneme analyzer (used by wav2vec handler)
from modules.phoneme_handler import PhonemeAnalyzer

# ---------------- Audio Recorder ----------------
from modules.recorder import AudioRecorder

# Tone Grader
from modules.tone_grader_word import ToneGraderWord, GradingResult

# Load vocabulary database
def load_mandarin_vocab(filename='mandarin_vocab.json'):
    """Load Mandarin vocabulary from JSON file"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data['words'])} words and {len(data['sentences'])} sentences")
        return data
    except FileNotFoundError:
        print(f"Vocab file {filename} not found, using defaults")
        return {"words": {}, "sentences": []}
    except Exception as e:
        print(f"Error loading vocab: {e}")
        return {"words": {}, "sentences": []}

# Load the database
VOCAB_DB = load_mandarin_vocab()

def get_word_info(word):
    """Get information about a Chinese word"""
    return VOCAB_DB['words'].get(word, None)

def get_sentence_by_id(sentence_id):
    """Get a sentence by its ID"""
    for sentence in VOCAB_DB['sentences']:
        if sentence['id'] == sentence_id:
            return sentence
    return None

def get_random_sentence():
    """Get a random sentence for practice"""
    import random
    if VOCAB_DB['sentences']:
        return random.choice(VOCAB_DB['sentences'])
    return None

def list_all_sentences():
    """List all available sentences"""
    return [(s['id'], s['mandarin'], s['english']) for s in VOCAB_DB['sentences']]

def set_target_word(word):
    """Set target word for practice"""
    info = get_word_info(word)
    if info:
        return {
            'character': word,
            'pinyin': info['pinyin'],
            'phonemes': info['phonemes'],
            'tone': info['tone'],
            'english': info['english']
        }
    return None

def get_random_practice_set_from_vocablist(vocab_list):
    """Get random practice items from VocabList instance"""
    import random
    
    practice_set = {
        'words': [],
        'sentences': []
    }
    
    # Separate words and sentences from vocab_list.all_items
    words = [item for item in vocab_list.all_items if item.get('type') == 'word']
    sentences = [item for item in vocab_list.all_items if item.get('type') == 'sentence']
    
    # Get 3 random words
    if len(words) >= 3:
        practice_set['words'] = random.sample(words, 3)
    else:
        practice_set['words'] = words
    
    # Get 2 random sentences
    if len(sentences) >= 2:
        practice_set['sentences'] = random.sample(sentences, 2)
    else:
        practice_set['sentences'] = sentences
    
    return practice_set

from pathlib import Path

class PracticeSession:
    """Manages practice session with multiple words and sentences"""
    
    def __init__(self, practice_set, audio_manager, audio_base_path='reference'):
        self.practice_set = practice_set
        self.audio_manager = audio_manager
        self.audio_base_path = Path(audio_base_path)
        self.practice_session = None
        self.current_index = 0
        self.all_items = practice_set['words']
        self.total_items = len(self.all_items)
    
    def get_current_item(self):
        """Get current practice item"""
        if 0 <= self.current_index < self.total_items:
            return self.all_items[self.current_index]
        return None
    
    def next_item(self):
        """Move to next item"""
        self.current_index = (self.current_index + 1) % self.total_items
        return self.get_current_item()
    
    def previous_item(self):
        """Move to previous item"""
        self.current_index = (self.current_index - 1) % self.total_items
        return self.get_current_item()
    
    def get_audio_for_current(self, voice_type='women'):
        """Get audio files for current item based on voice type"""
        item = self.get_current_item()
        if not item:
            return None, None
        
        item_id = item.get('id')
        if not item_id:
            print(f"Warning: Item missing 'id' field: {item}")
            return None, None
        
        # Construct path: reference/voice_type/id_voice_type.wav
        audio_path = self.audio_base_path / voice_type / f"{item_id}_{voice_type}.wav"
        
        if audio_path.exists():
            return str(audio_path), None
        else:
            print(f"Audio file not found: {audio_path}")
            return None, None
    
    def get_progress_string(self):
        """Get progress string like '2/5'"""
        return f"{self.current_index + 1} of {self.total_items}"
    
    def get_item_type(self):
        """Return 'word' or 'sentence'"""
        item = self.get_current_item()
        return 'word' if 'character' in item else 'sentence'

# Font setup
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

# Initialize font
font_path = get_font_path()

if font_path:
    font_prop = fm.FontProperties(fname=font_path, size=16)
    print(f"Using Doulos SIL font: {font_path}")
else:
    font_prop = fm.FontProperties(family='Times New Roman', size=16)
    print("Doulos SIL not found, using Times New Roman fallback")

# Wav2Vec2 imports
try:
    import torch
    import torchaudio
    WAV2VEC2_AVAILABLE = True
except ImportError:
    # print("Warning: torch/transformers not found. Install with: pip install torch torchaudio transformers")
    WAV2VEC2_AVAILABLE = False

# JAX/CARFAC imports
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

# ---------------- Chunk-based Wav2Vec2 Phoneme Handler ----------------
class SimpleWav2Vec2Handler:
    """Wav2Vec2 phoneme recognition handler - records all audio then processes once"""
    
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

        # Phoneme analyzer setup
        self.target_phonemes = target_phonemes
        self.phoneme_analyzer = PhonemeAnalyzer(self.target_phonemes)
        
        # Add phoneme analysis results storage
        self.analysis_results = None
        self.overall_score = 0.0

        # Callback system
        self.callbacks = []

        if self.enabled:
            try:
                self.microphone = sr.Microphone(sample_rate=self.sample_rate)
                self.recognizer = sr.Recognizer()
                with self.microphone:
                    self.recognizer.adjust_for_ambient_noise(self.microphone)
                print("🎤 Microphone ready")
            except Exception as e:
                print(f"⚠️ Microphone initialization warning: {e}. Disabling Wav2Vec2.")
                self.microphone = None
                self.recognizer = None
                self.enabled = False

    """
    def load_model(self):
        try:
            print(f"Loading Wav2Vec2 {self.model_name}...")
            # Load components (implementation details omitted for brevity)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.model_name)
            print("✅ Wav2Vec2 Model and Tokenizer loaded")
            return True
        except Exception as e:
            # print(f"Error loading model: {e}")
            return False
    """

    def get_current_status(self):
        """Simple 3-state status"""
        if not self.enabled:
            return "Wav2Vec2 not available", "red"
        
        if self.is_recording:
            return "Recording...", "yellow"
        
        if self.is_processing:
            return "Processing phonemes...", "orange"
        
        if self.result is not None:
            if self.result == "no_audio":
                return "No audio detected", "gray"
            else:
                return f"Transcription: {self.result[:20]}{'...' if len(self.result) > 20 else ''}", "black"
        
        return "Ready to start", "blue"

    def start_recording(self):
        """Start recording (continuous for the duration of the button press)"""
        # (Self-contained recording logic is moved to the SAIVisualizationWithWav2Vec2 class which uses the AudioRecorder module)
        # This handler will be used for post-recording analysis.
        pass

    def stop_recording(self):
        """Stop recording and process"""
        # (This is called by the SAIVisualizationWithWav2Vec2 after the recording is captured)
        pass

    def process_audio_buffer(self, complete_audio, target_phonemes):
        """Process an externally provided audio buffer with Wav2Vec2"""
        if not self.enabled:
            self.result = "Wav2Vec2 Disabled"
            return
            
        self.is_processing = True
        self.target_phonemes = target_phonemes
        self.analysis_results = None
        self.overall_score = 0.0
        self.result = None
        
        def _process():
            try:
                if complete_audio.size == 0:
                    self.result = "no_audio"
                    self.is_processing = False
                    return
                
                print(f"Processing audio chunk of length {complete_audio.size/self.sample_rate:.2f}s...")
                
                inputs = self.feature_extractor(
                    complete_audio, 
                    sampling_rate=self.sample_rate, 
                    return_tensors="pt"
                )
                
                with torch.no_grad():
                    logits = self.model(inputs.input_values).logits
                
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                cleaned_transcription = transcription.strip()
                
                if cleaned_transcription:
                    self.result = cleaned_transcription
                    
                    self.phoneme_analyzer.target_phonemes = target_phonemes
                    self.analysis_results, self.overall_score = self.phoneme_analyzer.analyze_pronunciation(
                        cleaned_transcription, self.target_phonemes
                    )
                    print(f"Phoneme analysis completed. Score: {self.overall_score:.2f}")
                else:
                    self.result = "no_audio"
                    print("No transcription generated")

                self.run_callbacks(complete_audio)
                    
            except Exception as processing_error:
                print(f"Processing error: {processing_error}")
                self.result = "processing_error"
                self.analysis_results = None
                self.overall_score = 0.0
                
            self.is_processing = False

        threading.Thread(target=_process, daemon=True).start()
        
    def get_current_result(self):
        """Get the latest transcription result"""
        return self.result if self.result else "no_audio"

    def register_callback(self, callback, *args):
        """Register a callback function to be called on audio processing completion"""
        self.callbacks.append((callback, args))

    def run_callbacks(self, complete_audio):
        """Run all registered callbacks"""
        for callback, args in self.callbacks:
            try:
                # Pass the raw audio, transcription, and score
                callback(complete_audio, self.result, self.overall_score, *args)
            except Exception as e:
                print(f"Callback error: {e}")

# ---------------- Audio Processor with Fallback ----------------
# (AudioProcessor and SAIProcessor classes are kept as in the provided code)
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
                # print("Using CARFAC audio processing")
            except Exception as e:
                # print(f"CARFAC initialization failed: {e}")
                self.use_carfac = False
                self.n_channels = 200
                # print("Falling back to simple numpy audio processor (CARFAC unavailable)")
        else:
            self.use_carfac = False
            self.n_channels = 200
            # print("JAX/CARFAC not available - using simple numpy audio processor")

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
                # print(f"CARFAC processing error: {e}")
                pass

        # Fallback simple processor when CARFAC/JAX isn't available or fails
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
            # print(f"Fallback audio processing error: {e}")
            return np.zeros((self.n_channels, 0), dtype=np.float32)

# ---------------- SAI Processor with Fallback ----------------
class SAIProcessor:
    def __init__(self, sai_params):
        self.sai_params = sai_params
        if JAX_AVAILABLE:
            try:
                self.sai = sai.SAI(sai_params)
                self.use_sai = True
                # print("Using SAI processing")
            except Exception as e:
                # print(f"SAI initialization failed: {e}")
                self.use_sai = False
        else:
            self.use_sai = False
        
        if not self.use_sai:
            # print("Using simple autocorrelation")
            pass
    
    def RunSegment(self, nap_output):
        if self.use_sai:
            try:
                return self.sai.RunSegment(nap_output)
            except Exception as e:
                # print(f"SAI processing error: {e}")
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

# ---------------- Helper Classes for Practice Mode ----------------
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

    def get_audio_for_word(self, word, voice_type, index):
        # Placeholder logic: relies on file existence checking in PracticeSession
        return None

    def get_audio_for_sentence(self, sentence_id, voice_type, index):
        # Placeholder logic
        return None

# ---------------- Main SAI Visualization with Wav2Vec2 and Practice Mode ----------------
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
            sai_width=400,  # Match both
            future_lags=399,  # sai_width - 1
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
        
        self.similarity_display = None
        self.similarity_rect = None
        
        # --- PRACTICE MODE INTEGRATION ---
        self.voice_selector = VoiceSelector()
        self.audio_manager = AudioManager()
        self.practice_session = None # Placeholder, set in main
        self.wav2vec2_handler = SimpleWav2Vec2Handler(sample_rate=sample_rate, target_phonemes=self.target_phonemes)
        self.wav2vec2_handler.register_callback(self._handle_processing_complete)
        
        # Simple local audio recorder (uses the imported module)
        self.is_recording_simple = False
        self.recorder = AudioRecorder(sample_rate=self.sample_rate)
        self.vocab_list = None  # Will be set in main()
        self.audio_base_path = None  # Will be set in main()
        # Tone grader (for words)
        self.grader = ToneGraderWord()

        # self.recorder.add_audio_callback(self._save_recording_with_metadata)
        self.recorder.add_audio_callback(self._grade_recording)

        self._setup_dual_visualization()

    def _play_audio_file(self, audio_data, sample_rate):
        """Play a numpy array of audio data"""
        if self.audio_output_stream and self.audio_output_stream.active:
            self.audio_output_stream.stop()
        
        if self.audio_data is not None and self.audio_output_stream:
            # Clear old stream
            self.audio_output_stream.close()

        if audio_data is None or audio_data.size == 0:
            print("No audio data to play.")
            self.audio_data = None
            self.total_samples = 0
            self.current_position = 0
            self.duration = 0
            self.playback_position = 0.0
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
        """Helper to format the display string for a word or sentence"""
        if not item:
            return "End of Set"
        if 'chinese' in item and item.get('type') == 'word':
            return f"WORD: {item['chinese']} ({item['pinyin']}) - {item['english']}"
        else:
            return f"SENTENCE: {item['chinese']} - {item['english']}"

    def _load_practice_item(self, item):
        """Loads the current item for practice"""
        if not item:
            return
        
        print(f"\n🎧 Loading item ({self.practice_session.get_progress_string()}): {self._get_item_display(item)}")
        
        # 1. Update text fields for the Reference SAI
        reference_pronunciation = item.get('pinyin', item.get('chinese'))
        translation = item.get('english')
        target_phonemes = item.get('phonemes', 'placeholder')
        
        self.set_reference_text(target_phonemes, reference_pronunciation, translation)
        self.wav2vec2_handler.target_phonemes = target_phonemes
        self.clear_phoneme_feedback()

        # 2. Load the reference audio and visualize its SAI
        audio_path, _ = self.practice_session.get_audio_for_current(self.voice_selector.current_voice)
        
        if audio_path and os.path.exists(audio_path):
            print(f"Loading reference audio: {audio_path}")
            
            # Load audio for SAI processing
            audio_data, original_sr = librosa.load(audio_path, sr=None)
            
            if original_sr != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=self.sample_rate)
            
            # Set the new audio data for the continuous visualization loop
            self.audio_data = audio_data
            self.total_samples = len(self.audio_data)
            self.duration = self.total_samples / self.sample_rate
            self.current_position = 0
            self.playback_position = 0.0
            
            # Also play the audio immediately if stream is active
            if self.audio_output_stream and self.audio_output_stream.active:
                self._play_audio_file(audio_data, self.sample_rate)

            # Clear and reset the file SAI visualization
            self.vis_file.img[:] = 0
            self.im_file.set_data(self.vis_file.img)
            self.fig.canvas.draw_idle()
            
        else:
            print(f"⚠️ Reference audio file not found for {reference_pronunciation} at path: {audio_path}")
            self.audio_data = None
            self.total_samples = 0
            self.duration = 0
            self.current_position = 0
            self.playback_position = 0.0

        if hasattr(self, 'practice_text'):
            item_text = f"[{item['type'].upper()}] {item['chinese']}\n{item['pinyin']}\n{item['english']}"
            self.practice_text.set_text(item_text)

        if hasattr(self, 'progress_text'):
            self.progress_text.set_text(self.practice_session.get_progress_string())
        
        # FIX: Update status to "Ready" instead of "Playing reference audio..."
        if hasattr(self, 'status_text'):
            self.status_text.set_text('Ready')
            self.status_text.set_color('lime')
            

    def clear_phoneme_feedback(self, event=None):
        """Clear the Wav2Vec2 and score feedback"""
        # Clear the user's SAI visualization
        self.vis_realtime.img[:] = 0
        self.im_realtime.set_data(self.vis_realtime.img)
        
        # Update status in practice area
        if hasattr(self, 'status_text'):
            self.status_text.set_text('Ready')
            self.status_text.set_color('lime')
        
        self.fig.canvas.draw_idle()

    def _handle_processing_complete(self, user_audio, transcription, score):
        """Callback run after Wav2Vec2 processing finishes"""
        print(f"Processing complete: Transcription='{transcription}', Score={score:.2f}")

        # 1. Update status text in practice area
        if hasattr(self, 'status_text'):
            if transcription and transcription != "no_audio":
                self.status_text.set_text(f'Score: {score:.1f}% | {transcription[:30]}...')
                self.status_text.set_color('yellow' if score < 70 else 'lime')
            else:
                self.status_text.set_text('No audio detected')
                self.status_text.set_color('orange')

        # 2. Re-process the user's audio through SAI/CARFAC for visualization
        self.vis_realtime.img[:] = 0
        self.im_realtime.set_data(self.vis_realtime.img)
        
        # Process user audio for SAI frame by frame
        total_frames = len(user_audio) // self.chunk_size
        remaining_samples = len(user_audio) % self.chunk_size
        
        def _process_user_sai():
            processor = AudioProcessor(self.sample_rate)
            sai_processor = SAIProcessor(self.sai_params)

            for i in range(total_frames):
                start = i * self.chunk_size
                end = start + self.chunk_size
                chunk = user_audio[start:end]
                
                nap_output = processor.process_chunk(chunk)
                sai_output = sai_processor.RunSegment(nap_output)
                
                self.vis_realtime.get_vowel_embedding(nap_output)
                self.vis_realtime.run_frame(sai_output)
                
                if self.vis_realtime.img.shape[1] > 1:
                    self.vis_realtime.img[:, :-1] = self.vis_realtime.img[:, 1:]
                    self.vis_realtime.draw_column(self.vis_realtime.img[:, -1])
                    
            if remaining_samples > 0:
                start = total_frames * self.chunk_size
                chunk = np.pad(user_audio[start:], (0, self.chunk_size - remaining_samples), 'constant')
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
        self.update_sai_speed_display()
        # print(f"SAI speed: {self.sai_speed:.1f}x")

    def increase_sai_speed(self, event=None):
        self.sai_speed = min(5.0, self.sai_speed + 0.25)
        self.update_sai_speed_display()
        # print(f"SAI speed: {self.sai_speed:.1f}x")

    def update_sai_speed_display(self):
        pass

    def decrease_audio_speed(self, event=None):
        self.playback_speed = max(0.25, self.playback_speed - 0.25)
        self.update_audio_speed_display()
        # print(f"Audio speed: {self.playback_speed:.1f}x")

    def increase_audio_speed(self, event=None):
        self.playback_speed = min(5.0, self.playback_speed + 0.25)
        self.update_audio_speed_display()
        # print(f"Audio speed: {self.playback_speed:.1f}x")

    def update_audio_speed_display(self):
        pass

    def toggle_voice(self, event=None):
        """Toggle reference voice between men/women"""
        new_voice = self.voice_selector.toggle()
        self.btn_voice.label.set_text(self.voice_selector.get_display_name())
        
        # Reload current item to update audio
        self._load_practice_item(self.practice_session.get_current_item())
        
        print(f"Reference voice switched to: {new_voice}")

    def on_key_press(self, event):
        if event.key == 'up' or event.key == '+':
            self.increase_sai_speed()
        elif event.key == 'down' or event.key == '-':
            self.decrease_sai_speed()
        elif event.key == 'right':
            self.increase_audio_speed()
        elif event.key == 'left':
            self.decrease_audio_speed()
        elif event.key == 'r':
            self.sai_speed = 1.0
            self.playback_speed = 1.0
            self.update_sai_speed_display()
            self.update_audio_speed_display()
            print("Both speeds reset to 1.0x")
        elif event.key == 'c':
            self.clear_phoneme_feedback()

    def _load_audio_file(self):
        # This is primarily used for the continuous SAI visualization for the reference audio
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
        # Note: No need to update text overlays - they're now in practice area


    def _setup_audio_playback(self):
        try:
            self.audio_output_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=self._audio_playback_callback
            )
            print("Audio playback stream created")
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
                
                # Check for wrap-around and looping
                if self.loop_audio and np.any(chunk_indices >= self.total_samples):
                    chunk_indices = chunk_indices % self.total_samples
                    
                chunk_indices = np.clip(chunk_indices, 0, self.total_samples - 1)
                chunk = self.audio_data[chunk_indices]

                outdata[:len(chunk), 0] = chunk
                outdata[len(chunk):, 0].fill(0) # Pad with zeros if chunk is shorter
                
                self.playback_position += int(frames * speed_factor)
                if self.playback_position >= self.total_samples and self.total_samples > 0:
                    if self.loop_audio:
                        self.playback_position = self.playback_position % self.total_samples
                    else:
                        outdata.fill(0)
                        # Stop the stream when finished if not looping
                        raise sd.CallbackStop # Signal stream to stop

            else:
                outdata.fill(0)
        except sd.CallbackStop:
            raise
        except Exception as e:
            # print(f"Audio callback error: {e}")
            outdata.fill(0)

    def get_next_file_chunk(self):
        if self.audio_data is None or self.total_samples == 0:
            return None, -1
        
        # Handle looping for the continuous SAI visualization stream
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
        """Setup PyAudio input stream for continuous SAI"""
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
        """Callback to feed audio queue for SAI processing"""
        try:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            if not self.audio_queue.full():
                self.audio_queue.put(audio_data)
        except Exception as e:
            pass
        return (in_data, pyaudio.paContinue)

    def process_realtime_audio(self):
        """Actually process the queue for SAI visualization"""
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Process through CARFAC and SAI
                nap_output = self.processor_realtime.process_chunk(audio_chunk)
                sai_output = self.sai_realtime.RunSegment(nap_output)
                self.vis_realtime.get_vowel_embedding(nap_output)
                self.vis_realtime.run_frame(sai_output)
                
                # Update visualization
                if self.vis_realtime.img.shape[1] > 1:
                    self.vis_realtime.img[:, :-1] = self.vis_realtime.img[:, 1:]
                    self.vis_realtime.draw_column(self.vis_realtime.img[:, -1])
                    
            except queue.Empty:
                continue

    def update_visualization(self, frame):
        try:
            # --- Reference SAI (File) Update ---
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
                                    
                    except Exception as e:
                        pass
                
                current_max_file = np.max(self.vis_file.img) if self.vis_file.img.size else 1
                self.im_file.set_data(self.vis_file.img)
                self.im_file.set_clim(vmin=0, vmax=max(1, min(255, current_max_file * 1.3)))

            # --- User SAI (Realtime) Update ---
            current_max_rt = np.max(self.vis_realtime.img) if self.vis_realtime.img.size > 0 else 1
            self.im_realtime.set_data(self.vis_realtime.img)
            self.im_realtime.set_clim(vmin=0, vmax=max(1, min(255, current_max_rt * 1.3)))

        except Exception as e:
            pass

        # Return only what exists
        return [self.im_realtime, self.im_file, self.status_text, self.practice_text, self.progress_text]

    def _save_recording_with_metadata(self, audio_data):
        """Save recorded audio and metadata to files"""
        try:
            # Create recordings directory if it doesn't exist
            save_dir = Path("recordings")
            save_dir.mkdir(exist_ok=True)
            
            # Generate timestamp and filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            wav_filename = f"sai_{timestamp}.wav"
            txt_filename = f"sai_{timestamp}.txt"
            
            wav_path = save_dir / wav_filename
            txt_path = save_dir / txt_filename
            
            # Save WAV file
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_normalized = audio_data / np.max(np.abs(audio_data)) * 0.95
            else:
                audio_normalized = audio_data
            
            # Convert to 16-bit integer
            audio_int16 = (audio_normalized * 32767).astype(np.int16)
            
            # Write WAV file
            with wave.open(str(wav_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            # Save metadata text file
            current_item = self.practice_session.get_current_item()
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Recording: {wav_filename}\n")
                f.write(f"Sample Rate: {self.sample_rate} Hz\n")
                f.write(f"Duration: {len(audio_data)/self.sample_rate:.2f} seconds\n")
                f.write(f"\n--- Practice Item ---\n")
                f.write(f"Item ID: {current_item.get('id', 'N/A')}\n")
                f.write(f"Type: {current_item.get('type', 'N/A')}\n")
                f.write(f"Chinese: {current_item.get('chinese', 'N/A')}\n")
                f.write(f"Pinyin: {current_item.get('pinyin', 'N/A')}\n")
                f.write(f"English: {current_item.get('english', 'N/A')}\n")
                f.write(f"Phonemes: {current_item.get('phonemes', 'N/A')}\n")
                f.write(f"\n--- Session Info ---\n")
                f.write(f"Progress: {self.practice_session.get_progress_string()}\n")
                f.write(f"Voice Type: {self.voice_selector.current_voice}\n")
                
                # Add Wav2Vec2 results if available
                if hasattr(self, 'wav2vec2_handler') and self.wav2vec2_handler.result:
                    f.write(f"\n--- Wav2Vec2 Analysis ---\n")
                    f.write(f"Transcription: {self.wav2vec2_handler.result}\n")
                    f.write(f"Score: {self.wav2vec2_handler.overall_score:.2f}%\n")
            
            print(f"Recording saved: {wav_path}")
            print(f"Metadata saved: {txt_path}")
            
            # Update status text
            if hasattr(self, 'status_text'):
                self.status_text.set_text(f'Saved: {wav_filename}')
                self.status_text.set_color('lime')
                self.fig.canvas.draw_idle()

            duration = len(audio_int16) / self.sample_rate
            print(f"Recording stopped - captured {duration:.1f} seconds.")
            
        except Exception as e:
            print(f"❌ Error saving recording: {e}")
            traceback.print_exc()
    
    def _grade_recording(self, audio_data):
        """Grade the recorded audio if it's a word"""
        current_item = self.practice_session.get_current_item()
        chinese = current_item.get('chinese', None)
        pinyin = current_item.get('pinyin', None)
        english = current_item.get('english', None)
        tones = current_item.get('tone', None)

        if type(tones) == list:
            tones = tones[0]

        print(audio_data, current_item)

        self.grader.grade_audio(audio_data, chinese, pinyin, english, tones)

        # Create recordings directory if it doesn't exist
        save_dir = Path("recordings")
        save_dir.mkdir(exist_ok=True)
            
        # Generate timestamp and filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        txt_filename = f"sai_{timestamp}.txt"

        self.grader.save_results(save_dir / txt_filename)


    def start(self):
        """Start the SAI visualization and main loop"""
        self.running = True
        
        # Initialize the audio playback stream (will be used by _play_audio_file)
        self._setup_audio_playback()
        
        # Start microphone stream for continuous SAI visualization
        self._setup_mic_stream()
        
        # Start real-time audio processing thread
        threading.Thread(target=self.process_realtime_audio, daemon=True).start()
        
        # Start the Matplotlib animation loop
        self.ani = animation.FuncAnimation(
            self.fig, self.update_visualization, 
            interval=int((self.chunk_size / self.sample_rate) * 1000), blit=True
        )
        print("Starting visualization...")
        plt.show()

    def stop(self):
        """Stop all processes and clean up"""
        self.running = False
        if self.audio_output_stream:
            self.audio_output_stream.stop()
            self.audio_output_stream.close()
        sd.stop()
        plt.close(self.fig)
        print("SAIVisualizationWithWav2Vec2 stopped.")

    def _setup_dual_visualization(self):
        """Create visualization with side-by-side SAI comparison - UPDATED TO MATCH REFERENCE"""
        self.fig = plt.figure(figsize=(16, 10))
        # Match reference: 3 rows with 7:2:0.3 ratio
        gs = self.fig.add_gridspec(3, 2, height_ratios=[7, 2, 0.3], width_ratios=[1, 1])

        # LEFT SAI display (Your Audio - Live) - SWAPPED to match reference
        self.ax_realtime = self.fig.add_subplot(gs[0, 0])
        self.im_realtime = self.ax_realtime.imshow(
            self.vis_realtime.img, aspect='auto', origin='lower',
            interpolation='bilinear', extent=[self.sai_width, 0, 0, self.n_channels],
            cmap='magma', vmin=0, vmax=255
        )
        self.ax_realtime.set_title('Your Audio (Live)', color='lime', fontsize=13, weight='bold')
        self.ax_realtime.axis('off')

        # RIGHT SAI display (Reference Audio) - SWAPPED to match reference
        self.ax_file = self.fig.add_subplot(gs[0, 1])
        self.im_file = self.ax_file.imshow(
            self.vis_file.img, aspect='auto', origin='lower',
            interpolation='bilinear', extent=[self.sai_width, 0, 0, self.n_channels],
            cmap='magma', vmin=0, vmax=255
        )
        self.ax_file.set_title('Reference Audio', color='cyan', fontsize=13, weight='bold')
        self.ax_file.axis('off')

        # Practice item display (spans both columns) - Row 1
        self.ax_practice = self.fig.add_subplot(gs[1, :])
        self.ax_practice.axis('off')
        self.ax_practice.set_facecolor('#16213e')

        current_item = self.practice_session.get_current_item() if self.practice_session else None
        if current_item:
            item_text = f"[{current_item['type'].upper()}] {current_item['chinese']}\n{current_item['pinyin']}\n{current_item['english']}"
        else:
            item_text = "Loading..."
        
        # Main practice text - centered
        self.practice_text = self.ax_practice.text(
            0.5, 0.5, item_text, transform=self.ax_practice.transAxes,
            color='cyan', fontsize=18, verticalalignment='center',
            horizontalalignment='center', weight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='black', alpha=0.9, edgecolor='cyan', linewidth=2)
        )

        # Status text (bottom left of practice area)
        self.status_text = self.ax_practice.text(
            0.02, 0.02, 'Ready - Press Play to hear reference', transform=self.ax_practice.transAxes,
            color='lime', fontsize=11, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8)
        )

        # Progress indicator (top right of practice area)
        if self.practice_session:
            progress_text = f"Set | {self.practice_session.get_progress_string()}"
        else:
            progress_text = "Practice Mode"
        
        self.progress_text = self.ax_practice.text(
            0.98, 0.98, progress_text, transform=self.ax_practice.transAxes,
            color='yellow', fontsize=10, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )

        # Control buttons (bottom row - Row 2) - Match reference spacing
        from matplotlib.widgets import Button

        self.ax_play_button = plt.axes([0.20, 0.08, 0.10, 0.05])
        self.btn_playback = Button(self.ax_play_button, 'Play',
                                color='lightcyan', hovercolor='cyan')
        self.btn_playback.on_clicked(self.toggle_playback)

        self.ax_rec_button = plt.axes([0.35, 0.08, 0.12, 0.05])
        self.btn_record = Button(self.ax_rec_button, 'Start Record',
                                color='lightgreen', hovercolor='green')
        self.btn_record.on_clicked(self.toggle_record)

        self.ax_save_button = plt.axes([0.53, 0.08, 0.12, 0.05])
        self.btn_save = Button(self.ax_save_button, 'Save Recording',
                            color='lightblue', hovercolor='blue')
        self.btn_save.on_clicked(self._save_recording_with_metadata)

        self.ax_next_button = plt.axes([0.70, 0.08, 0.10, 0.05])
        self.btn_next = Button(self.ax_next_button, 'Next Item',
                            color='lightyellow', hovercolor='yellow')
        self.btn_next.on_clicked(self.next_item)

        # Optional: Add "New Set" button if you want that feature
        # self.ax_newset_button = plt.axes([0.67, 0.08, 0.10, 0.05])
        # self.btn_newset = Button(self.ax_newset_button, 'New Set',
        #                          color='lightgray', hovercolor='gray')
        # self.btn_newset.on_clicked(self.generate_new_set)

        # Figure styling - match reference
        self.fig.patch.set_facecolor('#1a1a2e')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.96, bottom=0.06, hspace=0.15, wspace=0.15)
        
        # Key press handler
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # REMOVE these old text overlays on SAI images (they're now in practice area)
        # Delete or comment out:
        # self.transcription_realtime = self.ax_realtime.text(...)
        # self.transcription_file = self.ax_file.text(...)

    def generate_new_set(self, event=None):
        """Generate a completely new practice set"""
        if not self.vocab_list or not self.audio_base_path:
            print("ERROR: vocab_list or audio_base_path not initialized")
            return
        
        print("--- Generating New Practice Set ---")
        new_practice_set = get_random_practice_set_from_vocablist(self.vocab_list)
        self.practice_session = PracticeSession(new_practice_set, self.audio_manager, 
                                            audio_base_path=str(self.audio_base_path))
        self._load_practice_item(self.practice_session.get_current_item())
        self.progress_text.set_text(f"Set | {self.practice_session.get_progress_string()}")


    def toggle_record(self, event=None):
        """Toggle recording on/off"""
        if self.wav2vec2_handler.is_processing:
            print("⚠️ Cannot start recording: Wav2Vec2 is currently processing previous audio.")
            return

        if not self.is_recording_simple:
            # Start recording
            self.is_recording_simple = True
            self.btn_record.label.set_text('Recording...')
            self.btn_record.ax.set_facecolor('red')
            self.recorder.start_recording()
            self.clear_phoneme_feedback()
            
            if hasattr(self, 'status_text'):
                self.status_text.set_text('● Recording in progress...')
                self.status_text.set_color('red')  # FIX: Changed from = to ()
                self.fig.canvas.draw_idle()
            
            print("Recording started...")
        else:
            # Stop recording
            self.is_recording_simple = False
            self.btn_record.label.set_text('Start Record')
            self.btn_record.ax.set_facecolor('lightgreen')
            self.recorder.stop_recording()
            
            if hasattr(self, 'status_text'):
                self.status_text.set_text('Processing pronunciation...')
                self.status_text.set_color('orange')
            
            self.fig.canvas.draw_idle()

    def next_item(self, event=None):
        """Move to next practice item"""
        if self.practice_session:
            item = self.practice_session.next_item()
            if self.practice_session.current_index == 0:
                print("\nCompleted practice set! Starting over...")
            
            self._load_practice_item(item)
            
    def new_set(self, event=None):
        """Generate a new set of random practice items"""
        print("--- Generating New Practice Set ---")
        new_practice_set = get_random_practice_set()
        self.practice_session = PracticeSession(new_practice_set, self.audio_manager)
        
        # Load the first item
        self._load_practice_item(self.practice_session.get_current_item())
        print("Set loaded. Starting from the beginning.")

    def toggle_playback(self, event=None):
        """Play the reference audio for the current item"""
        if self.audio_output_stream and self.audio_output_stream.active:
            # If playing, stop it
            self.audio_output_stream.stop()
            self.btn_playback.label.set_text('Playing')
            print("Reference playback stopped")
        else:
            # If stopped, play the audio associated with the current item
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
            else:
                print("No practice item selected.")

# Add this class before the main() function
class VocabList:
    def __init__(self, audio_base_path="reference"):
        self.all_items = []
        self.audio_base_path = Path(audio_base_path)
        
        # 15 words - WAV format
        words = [
            {"type": "word", "id": 1, "chinese": "书", "pinyin": "shū", "english": "book", "phonemes": "ʂu", "tone": [1], "audio": "men/1_men.wav"},
            {"type": "word", "id": 2, "chinese": "女人", "pinyin": "nǚrén", "english": "woman", "phonemes": "ny ʐən", "tone": [3, 2], "audio": "women/2_women.wav"},
            {"type": "word", "id": 3, "chinese": "雄", "pinyin": "xióng", "english": "male/hero", "phonemes": "ɕjʊŋ", "tone": [1], "audio": "men/3_men.wav"},
            {"type": "word", "id": 4, "chinese": "去", "pinyin": "qù", "english": "to go", "phonemes": "tɕʰy", "tone": [4], "audio": "men/4_men.wav"},
            {"type": "word", "id": 6, "chinese": "喜欢", "pinyin": "xǐhuān", "english": "to like", "phonemes": "ɕi xwan", "tone": [3, 1], "audio": "women/6_women.wav"},
            {"type": "word", "id": 7, "chinese": "街道", "pinyin": "jiēdào", "english": "street", "phonemes": "tɕjɛ tɑʊ", "tone": [1, 4], "audio": "women/7_women.wav"},
            {"type": "word", "id": 8, "chinese": "熊猫", "pinyin": "xióngmāo", "english": "panda", "phonemes": "ɕjʊŋ mɑʊ", "tone": [2, 1], "audio": "men/8_men.wav"},
            {"type": "word", "id": 9, "chinese": "书店", "pinyin": "shūdiàn", "english": "bookstore", "phonemes": "ʂu tjɛn", "tone": [1, 4], "audio": "women/9_women.wav"},
            {"type": "word", "id": 10, "chinese": "去年", "pinyin": "qùnián", "english": "last year", "phonemes": "tɕʰy njɛn", "tone": [4, 2], "audio": "men/10_men.wav"},
            {"type": "word", "id": 11, "chinese": "中午", "pinyin": "zhōngwǔ", "english": "noon", "phonemes": "tʂʊŋ u", "tone": [1, 3], "audio": "women/11_women.wav"},
            {"type": "word", "id": 12, "chinese": "椅子", "pinyin": "yǐzi", "english": "chair", "phonemes": "i tsɿ", "tone": [3, 5], "audio": "men/12_men.wav"},
            {"type": "word", "id": 13, "chinese": "学校", "pinyin": "xuéxiào", "english": "school", "phonemes": "ɕɥɛ ɕjɑʊ", "tone": [2, 4], "audio": "women/13_women.wav"},
            {"type": "word", "id": 14, "chinese": "医院", "pinyin": "yīyuàn", "english": "hospital", "phonemes": "i ɥɛn", "tone": [1, 4], "audio": "men/14_men.wav"},
            {"type": "word", "id": 15, "chinese": "游戏", "pinyin": "yóuxì", "english": "game", "phonemes": "jɔʊ ɕi", "tone": [1, 4], "audio": "women/15_women.wav"},
            {"type": "word", "id": 16, "chinese": "她", "pinyin": "tā", "english": "she", "phonemes": "tʰa", "tone": [1], "audio": "men/16_men.wav"},
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
# ---------------- Main Entry Point ----------------
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="SAI Visualization and Mandarin Pronunciation Practice Tool.")
    parser.add_argument("--word", type=str, help="Specify a single Mandarin word for practice.")
    parser.add_argument("--sentence", type=int, help="Specify a sentence ID for practice.")
    args = parser.parse_args()

    # Remove this line:
    # from your_vocab_module import VocabList
    
    # Setup audio base path
    script_dir = Path(__file__).parent
    audio_base = script_dir / 'reference'
    
    # ... rest of main()
    
    # Initialize VocabList
    vocab_list = VocabList(audio_base_path=str(audio_base))
    
    if len(vocab_list.all_items) == 0:
        print("❌ No audio files found. Check your reference directory structure.")
        return 1

    practice_set = None
    audio_file_path = None
    word_info = None

    if args.word:
        # Find word in vocab_list by character
        word_info = next((item for item in vocab_list.all_items 
                         if item.get('type') == 'word' and item.get('chinese') == args.word), None)
        if word_info:
            # Use the audio path from VocabList
            audio_file_path = str(audio_base / word_info.get('audio'))
            print(f"Single word mode: {word_info['chinese']} ({word_info['pinyin']})")
        else:
            print(f"Word '{args.word}' not found in vocabulary.")
            return 1

    elif args.sentence:
        # Find sentence by ID
        word_info = next((item for item in vocab_list.all_items 
                         if item.get('type') == 'sentence' and item.get('id') == args.sentence), None)
        if word_info:
            audio_file_path = str(audio_base / word_info.get('audio'))
            word_info['phonemes'] = 'placeholder'
            print(f"Single sentence mode: {word_info['chinese']}")
        else:
            print(f"Sentence ID {args.sentence} not found.")
            return 1
    
    else:
        # Practice Mode
        print("--- Starting in Practice Mode (5 random items) ---")
        practice_set = get_random_practice_set_from_vocablist(vocab_list)
        if not practice_set['words'] and not practice_set['sentences']:
            print("❌ No practice items found.")
            return 1
        
        word_info = practice_set['words'][0] if practice_set['words'] else practice_set['sentences'][0]

    setup_chinese_font()
    
    try:
        sai_vis = SAIVisualizationWithWav2Vec2(
            audio_file_path=audio_file_path,
            playback_speed=1.0, 
            loop_audio=(practice_set is not None)
        )

        # ADD THESE LINES:
        sai_vis.vocab_list = vocab_list
        sai_vis.audio_base_path = audio_base
        
        if practice_set:
            # Pass audio_base_path to PracticeSession
            practice_session = PracticeSession(practice_set, sai_vis.audio_manager, 
                                              audio_base_path=str(audio_base))
            sai_vis.practice_session = practice_session
            sai_vis._load_practice_item(practice_session.get_current_item())
            
        elif word_info:
            # Map VocabList fields to expected fields
            reference_pronunciation = word_info.get('pinyin', word_info.get('chinese'))
            translation = word_info.get('english')
            target_phonemes = word_info.get('phonemes', 'placeholder')
            sai_vis.set_reference_text(target_phonemes, reference_pronunciation, translation)
            sai_vis.wav2vec2_handler.target_phonemes = target_phonemes
            sai_vis._load_audio_file()

        sai_vis.start()
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error starting visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if 'sai_vis' in locals():
            sai_vis.stop()
        print("✅ Visualization stopped cleanly")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())