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

# Configure matplotlib to support Chinese characters
def setup_chinese_font():
    """Setup matplotlib to display Chinese characters"""
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'STHeiti', 'Heiti TC',
        'Noto Sans CJK', 'WenQuanYi Micro Hei', 'Arial Unicode MS'
    ]

# ---------------- Visualization Handler ----------------
from modules.visualization_handler import VisualizationHandler, SAIParams

# Phoneme analyzer (used by wav2vec handler)
from modules.phoneme_handler import PhonemeAnalyzer

# ---------------- Audio Recorder ----------------
from modules.recorder import AudioRecorder

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

def get_random_practice_set():
    """Get a random set of 3 words and 2 sentences for practice"""
    import random
    
    practice_set = {
        'words': [],
        'sentences': []
    }
    
    # Get 3 random words
    all_words = list(VOCAB_DB['words'].keys())
    if len(all_words) >= 3:
        selected_words = random.sample(all_words, 3)
        for word in selected_words:
            info = VOCAB_DB['words'][word]
            practice_set['words'].append({
                'character': word,
                'pinyin': info['pinyin'],
                'phonemes': info['phonemes'],
                'tone': info['tone'],
                'english': info['english'],
                'id': info.get('id')
            })
    
    # Get 2 random sentences
    all_sentences = VOCAB_DB['sentences']
    if len(all_sentences) >= 2:
        selected_sentences = random.sample(all_sentences, 2)
        practice_set['sentences'] = selected_sentences
    
    return practice_set

class PracticeSession:
    """Manages practice session with multiple words and sentences"""
    
    def __init__(self, practice_set, audio_manager):
        self.practice_set = practice_set
        self.audio_manager = audio_manager
        self.practice_session = None  # Will be set if using practice mode
        self.current_index = 0
        self.all_items = practice_set['words'] + practice_set['sentences']
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
        """Get audio files for current item"""
        item = self.get_current_item()
        if not item:
            return None, None
        
        # Check if it's a word or sentence
        if 'character' in item:
            # It's a word
            # Note: Assuming audio paths are stored in the item structure or derived
            # For this example, we'll return a path if it exists, otherwise None.
            audio_path_key = f'{voice_type}_audio_path'
            audio_path = item.get(audio_path_key)
            
            # Simplified path generation for demonstration, assuming a structure
            if not audio_path:
                audio_path = os.path.join('audio', 'words', f"{item['character']}_{voice_type}.wav")
            
            return audio_path if os.path.exists(audio_path) else None, None
            
        else:
            # It's a sentence
            sentence_id = item.get('id')
            audio_path = os.path.join('audio', 'sentences', f"{sentence_id}_{voice_type}.wav")
            return audio_path if os.path.exists(audio_path) else None, None

    
    def get_progress_string(self):
        """Get progress string like '2/5'"""
        return f"{self.current_index + 1}/{self.total_items}"
    
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
    from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
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
                print("🎤 Microphone ready for Wav2Vec2")
            except Exception as e:
                print(f"⚠️ Microphone initialization warning: {e}. Disabling Wav2Vec2.")
                self.microphone = None
                self.recognizer = None
                self.enabled = False

            # Load model immediately
            if self.enabled and not self.load_model():
                print("❌ Failed to load Wav2Vec2 phoneme model. Handler disabled.")
                self.enabled = False
        else:
            print("⚠️ Wav2Vec2 disabled (missing dependencies)")

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
    def __init__(self, audio_file_path=None, chunk_size=1024, sample_rate=16000, sai_width=200,
                 debug=True, playback_speed=1.0, loop_audio=True):

        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sai_width = sai_width
        self.debug = debug
        self.playback_speed = playback_speed
        self.loop_audio = loop_audio
        self.sai_speed = 3.0
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
            sai_width=self.sai_width,
            future_lags=self.sai_width - 1,
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
        if 'character' in item:
            return f"WORD: {item['character']} ({item['pinyin']}) - {item['english']} ({item['tone']})"
        else:
            return f"SENTENCE: {item['mandarin']} - {item['english']}"

    def _load_practice_item(self, item):
        """Loads the current item for practice"""
        if not item:
            return
            
        print(f"\n🎧 Loading item ({self.practice_session.get_progress_string()}): {self._get_item_display(item)}")
        
        # 1. Update text fields for the Reference SAI
        reference_pronunciation = item.get('pinyin') if 'pinyin' in item else item.get('mandarin')
        translation = item.get('english')
        target_phonemes = item.get('phonemes')
        
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

    def clear_phoneme_feedback(self, event=None):
        """Clear the Wav2Vec2 and score feedback"""
        if hasattr(self, 'transcription_realtime'):
            self.transcription_realtime.set_text(self.wav2vec2_handler.get_current_status()[0])
            self.transcription_realtime.set_color(self.wav2vec2_handler.get_current_status()[1])
            self.vis_realtime.img[:] = 0 # Clear the user's SAI visualization
            self.im_realtime.set_data(self.vis_realtime.img)
            if hasattr(self, 'score_display'):
                self.score_display.set_text('Score: N/A')
            self.fig.canvas.draw_idle()

    def _handle_processing_complete(self, user_audio, transcription, score):
        """Callback run after Wav2Vec2 processing finishes"""
        print(f"Processing complete: Transcription='{transcription}', Score={score:.2f}")

        # 1. Update transcription/score text
        status_text, status_color = self.wav2vec2_handler.get_current_status()
        self.transcription_realtime.set_text(status_text)
        self.transcription_realtime.set_color(status_color)
        if hasattr(self, 'score_display'):
            self.score_display.set_text(f'Score: {score:.2f}' if score > 0 else 'Score: N/A')

        # 2. Re-process the user's audio through SAI/CARFAC for visualization
        self.vis_realtime.img[:] = 0 # Clear previous user SAI
        self.im_realtime.set_data(self.vis_realtime.img)
        
        # Process user audio for SAI frame by frame
        total_frames = len(user_audio) // self.chunk_size
        remaining_samples = len(user_audio) % self.chunk_size
        
        def _process_user_sai():
            processor = AudioProcessor(self.sample_rate) # New processor for the full buffer
            sai_processor = SAIProcessor(self.sai_params)

            for i in range(total_frames):
                start = i * self.chunk_size
                end = start + self.chunk_size
                chunk = user_audio[start:end]
                
                nap_output = processor.process_chunk(chunk)
                sai_output = sai_processor.RunSegment(nap_output)
                
                # Update visualization handler (simulating the realtime update)
                self.vis_realtime.get_vowel_embedding(nap_output)
                self.vis_realtime.run_frame(sai_output)
                
                # Shift and draw column
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
                    
            # Request redraw
            self.fig.canvas.draw_idle()

        # Run SAI processing in a separate thread so it doesn't block the main loop, 
        # as it can take a moment for the entire buffer.
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
        if hasattr(self, 'sai_speed_display'):
            self.sai_speed_display.set_text(f'SAI Speed: {self.sai_speed:.1f}x')

    def decrease_audio_speed(self, event=None):
        self.playback_speed = max(0.25, self.playback_speed - 0.25)
        self.update_audio_speed_display()
        # print(f"Audio speed: {self.playback_speed:.1f}x")

    def increase_audio_speed(self, event=None):
        self.playback_speed = min(5.0, self.playback_speed + 0.25)
        self.update_audio_speed_display()
        # print(f"Audio speed: {self.playback_speed:.1f}x")

    def update_audio_speed_display(self):
        if hasattr(self, 'audio_speed_display'):
            self.audio_speed_display.set_text(f'Audio Speed: {self.playback_speed:.1f}x')

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
        
        # Update the display text immediately
        if hasattr(self, 'transcription_file'):
            reference_display = f"Target: {self.reference_pronunciation}"
            if self.translated_text:
                reference_display += f" - {self.translated_text}"
            
            # Add progress string
            if hasattr(self, 'practice_session') and self.practice_session:
                 reference_display += f" ({self.practice_session.get_progress_string()})"
                 
            self.transcription_file.set_text(reference_display)
            self.fig.canvas.draw_idle()


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

    def process_realtime_audio(self):
        """Process real-time audio for SAI visualization only"""
        # print("Real-time SAI processing started")
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # We only display the reference SAI in this dual-view mode.
                # Real-time microphone audio is handled by the AudioRecorder module and Wav2Vec2.
                # This loop will now primarily run the SAI for the reference file.
                pass 
                
            except queue.Empty:
                continue
            except Exception as e:
                # print(f"Real-time processing error: {e}")
                continue

    def start(self):
        """Start the SAI visualization and main loop"""
        self.running = True
        
        # Initialize the audio playback stream (will be used by _play_audio_file)
        self._setup_audio_playback()
        
        # Start the Matplotlib animation loop
        self.ani = animation.FuncAnimation(
            self.fig, self.update_visualization, interval=int((self.chunk_size / self.sample_rate) * 1000), blit=True
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
        self.fig = plt.figure(figsize=(16, 14))
        # 12 rows for main content, 3 rows for controls (changed from 12 rows + controls)
        gs = self.fig.add_gridspec(15, 2, height_ratios=[1]*8 + [0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3]) 
        
        # --- Waveform Axis (Row 8-9) ---
        self.ax = self.fig.add_subplot(gs[8:10, 0])
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self._waveform_length = 256
        self._waveform_data = np.zeros(self._waveform_length, dtype=np.float32)
        self.line, = self.ax.plot(np.linspace(-1, 1, self._waveform_length), self._waveform_data, lw=2, color='skyblue')
        self.ax.axis('off')

        # --- SAI Axes (Row 0-7) ---
        # Left: Your Audio (Realtime)
        self.ax_realtime = self.fig.add_subplot(gs[0:8, 0])
        # Right: Reference Audio (File)
        self.ax_file = self.fig.add_subplot(gs[0:8, 1])

        self.im_realtime = self.ax_realtime.imshow(
            self.vis_realtime.img, aspect='auto', origin='upper', interpolation='bilinear', extent=[0, 200, 0, 200]
        )
        self.ax_realtime.axis('off')
        
        self.im_file = self.ax_file.imshow(
            self.vis_file.img, aspect='auto', origin='upper', interpolation='bilinear', extent=[0, 200, 0, 200]
        )
        self.ax_file.axis('off')
        
        # --- Text Overlays (Transcription/Status) ---
        # User Audio Status (Left)
        self.transcription_realtime = self.ax_realtime.text(
            0.02, 0.02, 'Live SAI', transform=self.ax_realtime.transAxes, verticalalignment='bottom', 
            fontsize=12, color='lime', weight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        # Reference Text (Right)
        self.transcription_file = self.ax_file.text(
            0.02, 0.02, '', transform=self.ax_file.transAxes, verticalalignment='bottom', 
            fontsize=12, color='cyan', weight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        # Score Display (Below User SAI)
        self.score_display = self.ax.text(
            0.5, 0.5, 'Score: N/A', transform=self.ax.transAxes, verticalalignment='center', horizontalalignment='center',
            fontsize=14, color='white', weight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.6)
        )

        # --- Control Axes (Row 10-14) ---
        self.ax_controls = self.fig.add_subplot(gs[10:14, :])
        self.ax_controls.axis('off')

        # Control layout parameters
        button_width = 0.10
        button_height = 0.04
        start_x = 0.05
        spacing = 0.02
        y_pos = 0.65 # Top row y position in the control axes

        # --- Control Buttons ---

        # 1. Play Reference Audio button
        self.ax_playback = plt.axes([start_x, y_pos, button_width, button_height])
        self.btn_playback = Button(self.ax_playback, '🔊 Play Ref', color='lightgreen', hovercolor='green')
        self.btn_playback.on_clicked(self.toggle_playback)

        # 2. Record button
        self.ax_record = plt.axes([start_x + (button_width + spacing), y_pos, button_width, button_height])
        self.btn_record = Button(self.ax_record, 'Start Record', color='lightcoral', hovercolor='red')
        self.btn_record.on_clicked(self.toggle_record)
        
        # 3. Next Item button
        self.ax_next = plt.axes([start_x + 2 * (button_width + spacing), y_pos, button_width, button_height])
        self.btn_next = Button(self.ax_next, 'Next Item', color='lightblue', hovercolor='blue')
        self.btn_next.on_clicked(self.next_item)

        # 4. New Set button
        self.ax_new_set = plt.axes([start_x + 3 * (button_width + spacing), y_pos, button_width, button_height])
        self.btn_new_set = Button(self.ax_new_set, 'New Set', color='lightgray', hovercolor='gray')
        self.btn_new_set.on_clicked(self.new_set)
        
        # 5. Voice Selector button
        self.ax_voice = plt.axes([start_x + 4 * (button_width + spacing), y_pos, button_width, button_height])
        self.btn_voice = Button(self.ax_voice, self.voice_selector.get_display_name(), color='orange', hovercolor='darkorange')
        self.btn_voice.on_clicked(self.toggle_voice)
        
        # 6. Clear Feedback button
        self.ax_clear = plt.axes([start_x + 5 * (button_width + spacing), y_pos, button_width, button_height])
        self.btn_clear = Button(self.ax_clear, 'Clear Feedback', color='gray', hovercolor='darkgray')
        self.btn_clear.on_clicked(self.clear_phoneme_feedback)

        # --- Speed Display Overlays (Below buttons) ---
        self.ax_speed_display = self.fig.add_subplot(gs[14, 0])
        self.ax_speed_display.axis('off')
        
        self.sai_speed_display = self.ax_speed_display.text(
            0.05, 0.5, f'SAI Speed: {self.sai_speed:.1f}x', transform=self.ax_speed_display.transAxes, 
            verticalalignment='center', horizontalalignment='left', fontsize=10, color='white'
        )
        self.audio_speed_display = self.ax_speed_display.text(
            0.5, 0.5, f'Audio Speed: {self.playback_speed:.1f}x', transform=self.ax_speed_display.transAxes, 
            verticalalignment='center', horizontalalignment='center', fontsize=10, color='white'
        )


        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.3)
        self.fig.patch.set_facecolor('black')
        
        # Bind keys to figure
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)


    def toggle_record(self, event=None):
        """Toggle recording on/off"""
        if self.wav2vec2_handler.is_processing:
            print("⚠️ Cannot start recording: Wav2Vec2 is currently processing previous audio.")
            return

        if not self.is_recording_simple:
            # Start recording
            self.is_recording_simple = True
            self.btn_record.label.set_text('🔴 Recording...')
            self.btn_record.color = 'red'
            self.btn_record.ax.set_facecolor('red')
            self.recorder.start_recording()
            self.clear_phoneme_feedback()
            print("Recording started...")
        else:
            # Stop recording
            self.is_recording_simple = False
            self.btn_record.label.set_text('Start Record')
            self.btn_record.color = 'lightcoral'
            self.btn_record.ax.set_facecolor('lightcoral')
            audio = self.recorder.stop_recording()
            
            # Update status immediately
            self.transcription_realtime.set_text("Processing...")
            self.transcription_realtime.set_color("orange")
            self.fig.canvas.draw_idle()
            
            print(f"Recording stopped - captured {len(audio)/self.sample_rate:.1f} seconds. Sending for analysis.")
            
            # Send audio buffer to Wav2Vec2 handler for analysis and callback
            self.wav2vec2_handler.process_audio_buffer(audio, self.target_phonemes)


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
            self.btn_playback.label.set_text('🔊 Play Ref')
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
                    self.btn_playback.label.set_text('⏹️ Stop Ref')
                else:
                    print(f"⚠️ Cannot play audio: File not found at {audio_path}")
            else:
                print("No practice item selected.")


    def update_visualization(self, frame):
        try:
            # Update the user's SAI visualization status (Wav2Vec2 status)
            if not self.wav2vec2_handler.is_processing and not self.is_recording_simple:
                status_text, status_color = self.wav2vec2_handler.get_current_status()
                # Only update if the handler status is not actively being changed by a callback
                if self.transcription_realtime.get_text() not in ("Processing...", "Recording..."):
                     self.transcription_realtime.set_text(status_text)
                     self.transcription_realtime.set_color(status_color)
            
            # --- Reference SAI (File) Update ---
            if self.audio_data is not None:
                chunk, chunk_index = self.get_next_file_chunk()
                if chunk is not None and chunk_index >= 0:
                    try:
                        nap_output = self.processor_file.process_chunk(chunk)
                        sai_output = self.sai_file.RunSegment(nap_output)
                        self.vis_file.get_vowel_embedding(nap_output)
                        self.vis_file.run_frame(sai_output)

                        # Logic to advance SAI visualization based on sai_speed
                        self.sai_file_index += self.sai_speed
                        if self.sai_file_index >= 1.0:
                            steps = int(self.sai_file_index)
                            self.sai_file_index -= steps
                            for _ in range(min(steps, 3)):
                                if self.vis_file.img.shape[1] > 1:
                                    self.vis_file.img[:, :-1] = self.vis_file.img[:, 1:]
                                    self.vis_file.draw_column(self.vis_file.img[:, -1])
                                    
                    except Exception as e:
                        # print(f"Error processing file chunk: {e}")
                        pass
                
                current_max_file = np.max(self.vis_file.img) if self.vis_file.img.size else 1
                self.im_file.set_data(self.vis_file.img)
                self.im_file.set_clim(vmin=0, vmax=max(1, min(255, current_max_file * 1.3)))
                
                # Update text on the right side with progress
                reference_display = f"Target: {self.reference_pronunciation}"
                if self.translated_text:
                    reference_display += f" - {self.translated_text}"
                reference_display += f" ({self.practice_session.get_progress_string()})"
                self.transcription_file.set_text(reference_display)


            # --- User SAI (Realtime) Update ---
            current_max_rt = np.max(self.vis_realtime.img) if self.vis_realtime.img.size > 0 else 1
            self.im_realtime.set_data(self.vis_realtime.img)
            self.im_realtime.set_clim(vmin=0, vmax=max(1, min(255, current_max_rt * 1.3)))
            
            # Update small waveform with most recent realtime img column
            if hasattr(self.vis_realtime, 'img') and self.vis_realtime.img is not None and self.vis_realtime.img.size > 0:
                col = self.vis_realtime.img[:, -1]
                col_mono = np.mean(col, axis=1) if col.ndim == 2 or col.ndim == 3 else col
                idx = np.linspace(0, len(col_mono) - 1, self._waveform_length).astype(int)
                data = col_mono[idx].astype(np.float32)
                self.line.set_data(np.linspace(-1, 1, len(data)), data)

        except Exception as e:
            # print(f"Visualization update error: {e}")
            pass

        # Return all mutable Matplotlib elements for blitting
        elements_to_return = [
            self.im_realtime, self.im_file, self.transcription_realtime, 
            self.transcription_file, self.line, self.score_display,
            self.sai_speed_display, self.audio_speed_display
        ]
        return [e for e in elements_to_return if e is not None]


# ---------------- Main Entry Point ----------------
def main():
    parser = argparse.ArgumentParser(description="SAI Visualization and Mandarin Pronunciation Practice Tool.")
    parser.add_argument("--word", type=str, help="Specify a single Mandarin word for practice.")
    parser.add_argument("--sentence", type=int, help="Specify a sentence ID for practice.")
    args = parser.parse_args()

    practice_set = None
    audio_file_path = None
    word_info = None

    if args.word:
        word_info = set_target_word(args.word)
        if word_info:
            audio_file_path = os.path.join('audio', 'words', f"{args.word}_women.wav")
            print(f"Single word mode: {word_info['character']} ({word_info['pinyin']})")
        else:
            print(f"Word '{args.word}' not found in vocabulary.")
            return 1

    elif args.sentence:
        word_info = get_sentence_by_id(args.sentence)
        if word_info:
            audio_file_path = os.path.join('audio', 'sentences', f"{args.sentence}_women.wav")
            # Sentence items use 'mandarin' and 'english' keys
            word_info['phonemes'] = 'placeholder' # Phoneme analysis must be provided
            print(f"Single sentence mode: {word_info['mandarin']}")
        else:
            print(f"Sentence ID {args.sentence} not found.")
            return 1
    
    else:
        # Default to Practice Mode if no arguments are provided
        print("--- Starting in Practice Mode (5 random items) ---")
        practice_set = get_random_practice_set()
        if not practice_set['words'] and not practice_set['sentences']:
            print("❌ No practice items found. Check 'mandarin_vocab.json'.")
            return 1
            
        # Select the first item to display initially
        word_info = practice_set['words'][0] if practice_set['words'] else practice_set['sentences'][0]
        # In practice mode, the first audio path is set when _load_practice_item is called in the main loop

    setup_chinese_font()
    
    try:
        # Initialize the main application
        sai_vis = SAIVisualizationWithWav2Vec2(
            audio_file_path=audio_file_path,
            playback_speed=1.0, 
            loop_audio=(practice_set is not None)
        )
        
        # Initialize the PracticeSession and load the first item
        if practice_set:
            practice_session = PracticeSession(practice_set, sai_vis.audio_manager)
            sai_vis.practice_session = practice_session
            sai_vis._load_practice_item(practice_session.get_current_item())
            
        # If in single-item mode, set the text directly and use the loaded audio
        elif word_info:
            reference_pronunciation = word_info.get('pinyin') if 'pinyin' in word_info else word_info.get('mandarin')
            translation = word_info.get('english')
            target_phonemes = word_info.get('phonemes')
            sai_vis.set_reference_text(target_phonemes, reference_pronunciation, translation)
            sai_vis.wav2vec2_handler.target_phonemes = target_phonemes
            sai_vis._load_audio_file() # Load the single file into the continuous SAI visualization

        
        sai_vis.start()
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
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