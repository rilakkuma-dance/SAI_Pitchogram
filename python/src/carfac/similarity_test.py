import sys
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button
import threading
import queue
import time
import torch
import whisper
from datetime import datetime, timedelta
from dataclasses import dataclass
import librosa
import soundfile as sf
import argparse
import os

import argostranslate.package
import argostranslate.translate

sys.path.append('./jax')
import jax
import jax.numpy as jnp
import carfac.jax.carfac as carfac

from carfac.np.carfac import CarParams
import sai

@dataclass
class SAIParams:
    """
    SAI Parameters
    """
    num_channels: int = 200
    sai_width: int = 400
    future_lags: int = 5
    num_triggers_per_frame: int = 10
    trigger_window_width: int = 20
    input_segment_width: int = 30
    channel_smoothing_scale: float = 0.5

    def __dict__(self):
        return {
            "num_channels": self.num_channels,
            "sai_width": self.sai_width,
            "future_lags": self.future_lags,
            "num_triggers_per_frame": self.num_triggers_per_frame,
            "trigger_window_width": self.trigger_window_width,
            "input_segment_width": self.input_segment_width,
            "channel_smoothing_scale": self.channel_smoothing_scale,
        }

@dataclass
class PitchogramParams:
    log_lag: bool = True
    lags_per_octave: float = 36.0
    min_lag_s: float = 0.0005
    log_offset_s: float = 0.0025
    vowel_time_constant_s: float = 0.02
    light_theme: bool = False

# ---------------- Similarity Calculator ----------------
class SimilarityCalculator:
    """
    Calculates similarity scores between real-time and recorded voice SAI representations
    """
    def __init__(self, history_size=50, smoothing_factor=0.3):
        self.history_size = history_size
        self.smoothing_factor = smoothing_factor
        
        # Circular buffers for storing recent SAI frames
        self.realtime_history = []
        self.recorded_history = []
        
        # Similarity metrics
        self.current_similarity = 0.0
        self.smoothed_similarity = 0.0
        self.max_similarity = 0.0
        self.similarity_history = []
        
        # Feature comparison methods
        self.comparison_methods = {
            'cosine': self._cosine_similarity,
            'correlation': self._correlation_similarity,
            'euclidean': self._euclidean_similarity,
            'spectral': self._spectral_similarity
        }
        
    def add_realtime_frame(self, sai_frame):
        """Add a real-time SAI frame to history"""
        if len(self.realtime_history) >= self.history_size:
            self.realtime_history.pop(0)
        self.realtime_history.append(sai_frame.copy())
    
    def add_recorded_frame(self, sai_frame):
        """Add a recorded SAI frame to history"""
        if len(self.recorded_history) >= self.history_size:
            self.recorded_history.pop(0)
        self.recorded_history.append(sai_frame.copy())
    
    def calculate_similarity(self, method='cosine', temporal_window=5):
        """
        Calculate similarity between recent real-time and recorded SAI frames
        
        Args:
            method: Similarity calculation method ('cosine', 'correlation', 'euclidean', 'spectral')
            temporal_window: Number of recent frames to compare
        """
        if len(self.realtime_history) == 0 or len(self.recorded_history) == 0:
            return 0.0
        
        # Get recent frames
        rt_frames = self.realtime_history[-temporal_window:]
        rec_frames = self.recorded_history[-temporal_window:]
        
        if len(rt_frames) == 0 or len(rec_frames) == 0:
            return 0.0
        
        # Calculate similarity using specified method
        similarity_func = self.comparison_methods.get(method, self._cosine_similarity)
        similarities = []
        
        # Compare each real-time frame with each recorded frame in the temporal window
        for rt_frame in rt_frames:
            for rec_frame in rec_frames:
                sim = similarity_func(rt_frame, rec_frame)
                similarities.append(sim)
        
        # Take the maximum similarity (best match)
        if similarities:
            raw_similarity = max(similarities)
        else:
            raw_similarity = 0.0
        
        # Update current similarity
        self.current_similarity = raw_similarity
        
        # Apply exponential smoothing
        self.smoothed_similarity = (self.smoothing_factor * raw_similarity + 
                                   (1 - self.smoothing_factor) * self.smoothed_similarity)
        
        # Track maximum similarity
        self.max_similarity = max(self.max_similarity, raw_similarity)
        
        # Add to history
        if len(self.similarity_history) >= self.history_size:
            self.similarity_history.pop(0)
        self.similarity_history.append(raw_similarity)
        
        return self.smoothed_similarity
    
    def _cosine_similarity(self, frame1, frame2):
        """Calculate cosine similarity between two SAI frames"""
        try:
            # Flatten frames
            f1 = frame1.flatten()
            f2 = frame2.flatten()
            
            # Remove zero vectors
            if np.linalg.norm(f1) == 0 or np.linalg.norm(f2) == 0:
                return 0.0
            
            # Calculate cosine similarity
            dot_product = np.dot(f1, f2)
            norms = np.linalg.norm(f1) * np.linalg.norm(f2)
            
            if norms == 0:
                return 0.0
            
            similarity = dot_product / norms
            return max(0.0, similarity)  # Clamp to [0, 1]
            
        except Exception:
            return 0.0
    
    def _correlation_similarity(self, frame1, frame2):
        """Calculate correlation similarity between two SAI frames"""
        try:
            f1 = frame1.flatten()
            f2 = frame2.flatten()
            
            if len(f1) != len(f2) or len(f1) < 2:
                return 0.0
            
            correlation = np.corrcoef(f1, f2)[0, 1]
            
            if np.isnan(correlation):
                return 0.0
            
            return max(0.0, correlation)  # Clamp to [0, 1]
            
        except Exception:
            return 0.0
    
    def _euclidean_similarity(self, frame1, frame2):
        """Calculate normalized inverse Euclidean distance"""
        try:
            f1 = frame1.flatten()
            f2 = frame2.flatten()
            
            if len(f1) != len(f2):
                return 0.0
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(f1 - f2)
            
            # Convert to similarity (inverse relationship)
            # Normalize by the maximum possible distance
            max_distance = np.linalg.norm(np.ones_like(f1))
            if max_distance == 0:
                return 0.0
            
            similarity = 1.0 - (distance / max_distance)
            return max(0.0, similarity)
            
        except Exception:
            return 0.0
    
    def _spectral_similarity(self, frame1, frame2):
        """Calculate similarity based on spectral features"""
        try:
            # Take mean across channels for each lag (frequency analysis)
            spec1 = np.mean(frame1, axis=0)
            spec2 = np.mean(frame2, axis=0)
            
            # Calculate spectral correlation
            if len(spec1) < 2 or len(spec2) < 2:
                return 0.0
            
            correlation = np.corrcoef(spec1, spec2)[0, 1]
            
            if np.isnan(correlation):
                return 0.0
            
            return max(0.0, correlation)
            
        except Exception:
            return 0.0
    
    def get_similarity_stats(self):
        """Get similarity statistics"""
        if not self.similarity_history:
            return {
                'current': 0.0,
                'smoothed': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'std': 0.0
            }
        
        return {
            'current': self.current_similarity,
            'smoothed': self.smoothed_similarity,
            'max': self.max_similarity,
            'mean': np.mean(self.similarity_history),
            'std': np.std(self.similarity_history)
        }
    
    def reset(self):
        """Reset similarity calculator"""
        self.realtime_history.clear()
        self.recorded_history.clear()
        self.current_similarity = 0.0
        self.smoothed_similarity = 0.0
        self.max_similarity = 0.0
        self.similarity_history.clear()

# ---------------- CARFAC Processor ----------------
class RealCARFACProcessor:
    def __init__(self, fs=16000):
        self.fs = fs
        self.hypers, self.weights, self.state = carfac.design_and_init_carfac(
            carfac.CarfacDesignParameters(fs=fs, n_ears=1)
        )
        self.n_channels = self.hypers.ears[0].car.n_ch
        self.run_segment_jit = jax.jit(carfac.run_segment, static_argnames=['hypers', 'open_loop'])

    def process_chunk(self, audio_chunk):
        if len(audio_chunk.shape) == 1:
            audio_input = audio_chunk.reshape(-1, 1)
        else:
            audio_input = audio_chunk
        audio_jax = jnp.array(audio_input, dtype=jnp.float32)
        naps, _, self.state, _, _, _ = self.run_segment_jit(audio_jax, self.hypers, self.weights, self.state, open_loop=False)
        return np.array(naps[:, :, 0]).T

# ---------------- Whisper Handler ----------------
class WhisperHandler:
    def __init__(self, model_name="base", non_english=False, debug=True):
        self.debug = debug
        model = model_name
        if model_name != "large" and not non_english:
            model = model + ".en"
        
        try:
            self.audio_model = whisper.load_model(model)
            self.sample_rate = 16000
        except Exception as e:
            try:
                self.audio_model = whisper.load_model("tiny")
                self.sample_rate = 16000
            except Exception as e2:
                self.audio_model = None
                self.sample_rate = 16000
        
        self.transcription = []
        self.lock = threading.Lock()
        self.last_transcription_time = time.time()
        self.min_transcription_interval = 0.1

    def transcribe_audio(self, audio_data, language='en'):
        min_samples = int(self.sample_rate * 0.5)
        
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)
        
        max_val = np.abs(audio_float).max()
        if max_val > 1.0:
            audio_float = audio_float / max_val
        
        target_length = max(min_samples, len(audio_float))
        if len(audio_float) < target_length:
            audio_float = np.pad(audio_float, (0, target_length - len(audio_float)), 'constant')
        elif len(audio_float) > self.sample_rate * 30:
            audio_float = audio_float[-self.sample_rate * 30:]
        
        try:
            result = self.audio_model.transcribe(
                audio_float, 
                fp16=torch.cuda.is_available(),
                language=language,
                condition_on_previous_text=False
            )
            text = result.get('text', '').strip()
            
            if len(text) < 1:
                return None
                
            return text
        except Exception:
            return None

    def add_transcription_line(self, text):
        with self.lock:
            if text is None or not text.strip():
                return
            
            current_time = time.time()
            if current_time - self.last_transcription_time < self.min_transcription_interval:
                return
            
            if self.transcription and self.transcription[-1] == text:
                return
                
            self.transcription.append(text)
            self.last_transcription_time = current_time
            print(f"[Transcribed]: {text}")
            
            if len(self.transcription) > 20:
                self.transcription = self.transcription[-20:]

    def get_display_text(self, max_lines=5, max_chars=200):
        with self.lock:
            if not self.transcription:
                return ""
            
            lines = self.transcription[-max_lines:]
            display = '\n'.join(lines)
            
            if len(display) > max_chars:
                display = "..." + display[-(max_chars-3):]
            
            return display

# ---------------- Visualization Handler ----------------
class VisualizationHandler:
    @dataclass
    class ResamplingCell:
        left_index: int
        right_index: int
        left_weight: float
        interior_weight: float
        right_weight: float

        def __init__(self, left_edge: float, right_edge: float):
            cell_width: float = right_edge - left_edge

            if (cell_width < 1.0):
                grow: float = 0.5 * (1.0 - cell_width)
                left_edge -= grow
                right_edge += grow
            
            left_edge = max(0.0, left_edge)
            right_edge = max(0.0, right_edge)
            cell_width = right_edge - left_edge

            self.left_index = int(round(left_edge))
            self.right_index = int(round(right_edge))
            if (self.right_index > self.left_index and cell_width > 0.999):
                self.left_weight = (0.5 - (left_edge - self.left_index)) / cell_width
                self.interior_weight = 1.0 / cell_width
                self.right_weight = (0.5 + (right_edge - self.right_index)) / cell_width
            else:
                self.left_weight = 1.0
                self.interior_weight = 0.0
                self.right_weight = 0.0

        def CellAverage(self, samples: np.ndarray) -> float:
            if (self.left_index == self.right_index):
                return samples[self.left_index]
            return self.left_weight * samples[self.left_index] + \
                self.interior_weight * samples[self.left_index + 1 : self.right_index].sum() + \
                self.right_weight * samples[self.right_index]

    def __init__(self, sample_rate_hz: int, car_params: CarParams = CarParams(), sai_params: SAIParams = SAIParams(),
                 pitchogram_params: PitchogramParams = PitchogramParams()):
        self.temporal_buffer = np.zeros((sai_params.num_channels, sai_params.sai_width))

        self.car_params = car_params
        self.sai_params = sai_params
        self.pitchogram_params = pitchogram_params
        self.sample_rate_hz = sample_rate_hz

        self.workspace = np.zeros((sai_params.sai_width))
        self.output = np.zeros((sai_params.sai_width))

        self.pole_frequencies = self.car_pole_frequencies(sample_rate_hz, car_params)

        self.mask = np.ones((sai_params.num_channels, sai_params.sai_width), dtype=bool)
        center: int = sai_params.sai_width - sai_params.future_lags
        for c in range(self.pole_frequencies.shape[0]):
            half_cycle_samples: float = 0.5 * sample_rate_hz / self.pole_frequencies[c]
            i_start: int = int(np.clip(np.floor(center - half_cycle_samples), 0, sai_params.sai_width - 1))
            i_end: int = int(np.clip(np.floor(center + half_cycle_samples), 0, sai_params.sai_width - 1))
            self.mask[c, i_start:i_end+1] = 0

        self.vowel_matrix = self.create_vowel_matrix(sai_params.num_channels)
        self.vowel_coords = np.zeros((2, 1), dtype=np.float32)

        frame_rate_hz: float = sample_rate_hz / sai_params.input_segment_width
        self.cgram_smoother = 1 - np.exp(-1 / (pitchogram_params.vowel_time_constant_s * frame_rate_hz))
        self.cgram = np.zeros(self.pole_frequencies.shape, dtype=np.float32)

        self.log_lag_cells: list[VisualizationHandler.ResamplingCell] = list()
        if (not pitchogram_params.log_lag):
            self.output.resize((sai_params.sai_width))
        else:
            spacing: float = np.exp2(1.0 / pitchogram_params.lags_per_octave)
            log_offset: float = sample_rate_hz * pitchogram_params.log_offset_s
            left_edge: float = sample_rate_hz * pitchogram_params.min_lag_s

            while True:
                right_edge: float = (left_edge + log_offset) * spacing - log_offset
                cell: VisualizationHandler.ResamplingCell = self.ResamplingCell(left_edge, right_edge)
                if (cell.right_index >= sai_params.sai_width):
                    break
                self.log_lag_cells.append(cell)
                left_edge = right_edge

            self.workspace.resize((sai_params.sai_width))
            self.output.resize((len(self.log_lag_cells)))

        # Initialize visualization image
        self.img = np.zeros((self.output.shape[0], 200, 3), dtype=np.uint8)

    def car_pole_frequencies(self, sample_rate_hz, car_params: CarParams) -> np.ndarray:
        num_channels: int = 0
        pole_hz: float = car_params.first_pole_theta * sample_rate_hz / (2.0 * np.pi)
        while pole_hz > car_params.min_pole_hz:
            num_channels += 1
            pole_hz -= car_params.erb_per_step * \
                ((car_params.erb_break_freq + pole_hz) / car_params.erb_q)

        pole_freqs = np.zeros(num_channels, dtype=np.float32)
        pole_hz = car_params.first_pole_theta * sample_rate_hz / (2.0 * np.pi)
        for channel in range(num_channels):
            pole_freqs[channel] = pole_hz
            pole_hz -= car_params.erb_per_step * \
                ((car_params.erb_break_freq + pole_hz) / car_params.erb_q)
        return pole_freqs
    
    def create_vowel_matrix(self, num_channels, erb_per_step=0.5) -> np.ndarray:
        def kernel(center, c):
            z = (c - center) / 3.3
            return np.exp((z * z) / -2)

        f2_hi = self.frequency_to_channel_index(self.sample_rate_hz, erb_per_step, 2365)
        f2_lo = self.frequency_to_channel_index(self.sample_rate_hz, erb_per_step, 1100)
        f1_hi = self.frequency_to_channel_index(self.sample_rate_hz, erb_per_step, 700)
        f1_lo = self.frequency_to_channel_index(self.sample_rate_hz, erb_per_step, 265)

        vowel_matrix = np.zeros((2, num_channels), dtype=np.float32)
        for c in range(num_channels):
            vowel_matrix[0, c] = kernel(f2_lo, c) - kernel(f2_hi, c)
            vowel_matrix[1, c] = kernel(f1_lo, c) - kernel(f1_hi, c)
        vowel_matrix *= erb_per_step / 2
        return vowel_matrix

    def frequency_to_channel_index(self, sample_rate_hz: int, erb_per_step: float, pole_freq: int):
        first_pole_theta: float = 0.85 * np.pi
        erb_q: float = 1000 / (24.7 * 4.37)

        pole0_hz: float = first_pole_theta * sample_rate_hz / (2.0 * np.pi)
        break_freq: float = 165.3
        ratio: float = 1 - erb_per_step / erb_q
        min_pole_hz: float = 30
        pole_freq = np.clip(pole_freq, min_pole_hz, pole0_hz)
        top = np.log((pole_freq + break_freq) / (pole0_hz + break_freq))
        bottom = np.log(ratio)
        return top / bottom

    def get_vowel_embedding(self, nap) -> np.ndarray:
        self.cgram += self.cgram_smoother * (nap.mean(axis=1) - self.cgram)
        self.vowel_coords = self.vowel_matrix @ self.cgram
        return self.vowel_coords

    def run_frame(self, sai_frame: np.ndarray) -> np.ndarray:
        if (not self.pitchogram_params.log_lag):
            self.output = (sai_frame * self.mask).mean(axis=0)
        else:
            self.workspace = (sai_frame * self.mask).mean(axis=0)
            for i in range(self.output.shape[0]):
                self.output[i] = self.log_lag_cells[i].CellAverage(self.workspace)

        return self.output

    def draw_column(self, column_ptr: np.ndarray) -> None:
        v = np.ravel(self.vowel_coords)
        tint = np.array([
            0.5 - 0.6 * v[1],
            0.5 - 0.6 * v[0],
            0.35 * (v[0] + v[1]) + 0.4
        ], dtype=np.float32)

        k_scale: float = 0.5 * 255
        tint *= k_scale

        for i in range(self.output.shape[0]):
            column_ptr[i] = np.clip(np.int32((tint * self.output[i])), 0, 255)

# ---------------- Voice-to-Voice SAI Processor ----------------
class VoiceSimilarityProcessor:
    def __init__(self, chunk_size=1024, sample_rate=16000, sai_width=200,
                 whisper_model="tiny", whisper_interval=1.5, 
                 debug=True, similarity_method='cosine'):
        
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sai_width = sai_width
        self.whisper_interval = whisper_interval
        self.debug = debug
        
        # Initialize similarity calculator
        self.similarity_calculator = SimilarityCalculator(
            history_size=50,
            smoothing_factor=0.2
        )
        self.similarity_method = similarity_method
        self.similarity_display_history = []
        self.max_similarity_display_history = 200
        
        # Recording functionality
        self.is_recording = False
        self.recorded_audio = []
        self.recorded_sai_frames = []
        self.recording_start_time = 0
        self.recording_duration = 5.0
        self.last_score_percentage = 0.0
        self.score_history = []
        
        # Real-time SAI buffer for comparison (circular buffer)
        self.realtime_sai_buffer = []
        self.max_realtime_buffer_size = 100
        
        # Initialize processing components
        self.carfac_realtime = RealCARFACProcessor(fs=sample_rate)
        self.n_channels = self.carfac_realtime.n_channels

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
        
        # Create SAI instance
        self.SAI_realtime = sai.SAI(self.sai_params)

        # Visualization handler
        self.vis_realtime = VisualizationHandler(sample_rate, sai_params=self.sai_params)

        # Whisper handler
        self.whisper_realtime = WhisperHandler(model_name=whisper_model, debug=debug)

        # Real-time audio setup
        self.audio_queue = queue.Queue(maxsize=50)
        self.whisper_audio_buffer_realtime = []
        self.whisper_buffer_lock_realtime = threading.Lock()
        self.last_whisper_time_realtime = time.time()

        # Audio stream
        self.p = None
        self.stream = None
        self.running = False
        
        # Setup visualization
        self._setup_visualization()

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Real-time audio callback with recording capability"""
        try:
            audio_float = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Handle recording
            if self.is_recording:
                self.recorded_audio.extend(audio_float)
                # Check if recording duration exceeded
                if len(self.recorded_audio) >= int(self.recording_duration * self.sample_rate):
                    self.stop_recording()
            
            try:
                self.audio_queue.put_nowait(audio_float)
            except queue.Full:
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_float)
                except queue.Empty:
                    pass

            # Add to Whisper buffer
            with self.whisper_buffer_lock_realtime:
                self.whisper_audio_buffer_realtime.extend(audio_float)
                
                max_buffer_size = self.sample_rate * 15
                if len(self.whisper_audio_buffer_realtime) > max_buffer_size:
                    excess = len(self.whisper_audio_buffer_realtime) - max_buffer_size
                    self.whisper_audio_buffer_realtime = self.whisper_audio_buffer_realtime[excess:]

        except Exception as e:
            print(f"Audio callback error: {e}")
        
        return (in_data, pyaudio.paContinue)

    def start_recording(self, event=None):
        """Start recording user's speech"""
        if self.is_recording:
            self.stop_recording()
            return
        
        self.is_recording = True
        self.recorded_audio = []
        self.recorded_sai_frames = []
        self.recording_start_time = time.time()
        
        # Update button appearance
        self.btn_record.label.set_text('⏹ REC')
        self.btn_record.color = (1, 0, 0, 0.8)  # Red when recording
        
        print(f"Recording started... Speak for {self.recording_duration} seconds")
        print("Your speech will be compared against recent real-time audio")
        
        # Start countdown thread
        threading.Thread(target=self._recording_countdown, daemon=True).start()

    def stop_recording(self, event=None):
        """Stop recording and calculate similarity score"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Update button appearance
        self.btn_record.label.set_text('🎤 REC')
        self.btn_record.color = (0, 0.8, 1, 0.8)  # Blue when not recording
        
        if len(self.recorded_audio) > 0:
            # Process recorded audio and calculate score
            threading.Thread(target=self._process_recorded_audio, daemon=True).start()
        else:
            print("No audio recorded")

    def _recording_countdown(self):
        """Handle recording countdown"""
        start_time = time.time()
        while self.is_recording and (time.time() - start_time) < self.recording_duration:
            remaining = self.recording_duration - (time.time() - start_time)
            if remaining > 0:
                time.sleep(0.1)
        
        if self.is_recording:
            self.stop_recording()

    def _process_recorded_audio(self):
        """Process recorded audio and compare against real-time SAI buffer"""
        try:
            print("Processing recorded audio...")
            
            # Convert to numpy array and ensure correct format
            recorded_array = np.array(self.recorded_audio, dtype=np.float32)
            
            if len(recorded_array) < self.sample_rate * 0.5:  # Less than 0.5 seconds
                print("Recording too short for analysis")
                return
            
            # Normalize audio
            if np.max(np.abs(recorded_array)) > 0:
                recorded_array = recorded_array / np.max(np.abs(recorded_array)) * 0.9
            
            # Process recorded audio through CARFAC and SAI
            temp_carfac = RealCARFACProcessor(fs=self.sample_rate)
            temp_sai = sai.SAI(self.sai_params)
            
            # Split into chunks and process
            chunk_size = self.chunk_size
            recorded_sai_frames = []
            
            for i in range(0, len(recorded_array), chunk_size):
                chunk = recorded_array[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
                
                # Process chunk
                nap_output = temp_carfac.process_chunk(chunk)
                sai_output = temp_sai.RunSegment(nap_output)
                recorded_sai_frames.append(sai_output)
            
            self.recorded_sai_frames = recorded_sai_frames
            
            # Compare against real-time SAI buffer instead of file
            if len(self.realtime_sai_buffer) > 0 and len(recorded_sai_frames) > 0:
                similarities = []
                
                # Get the most recent real-time frames (same number as recorded frames)
                num_frames_to_compare = min(len(recorded_sai_frames), len(self.realtime_sai_buffer))
                recent_realtime_frames = self.realtime_sai_buffer[-num_frames_to_compare:]
                
                # Calculate frame-by-frame similarities
                for i in range(num_frames_to_compare):
                    rec_frame = recorded_sai_frames[i]
                    rt_frame = recent_realtime_frames[i] if i < len(recent_realtime_frames) else recent_realtime_frames[-1]
                    
                    sim = self.similarity_calculator._cosine_similarity(rec_frame, rt_frame)
                    similarities.append(sim)
                
                if similarities:
                    # Calculate percentage score
                    avg_similarity = np.mean(similarities)
                    max_similarity = np.max(similarities)
                    
                    # Convert to percentage (0-100%)
                    percentage_score = avg_similarity * 100
                    max_percentage = max_similarity * 100
                    
                    self.last_score_percentage = percentage_score
                    self.score_history.append(percentage_score)
                    
                    # Keep only last 10 scores
                    if len(self.score_history) > 10:
                        self.score_history = self.score_history[-10:]
                    
                    # Provide feedback
                    self._provide_voice_feedback(percentage_score, max_percentage)
                    
                    print(f"Voice Similarity Score: {percentage_score:.1f}% (Peak: {max_percentage:.1f}%)")
                    print(f"Compared {len(similarities)} frame pairs between recorded and recent real-time audio")
                else:
                    print("Could not calculate similarity - no matching frames")
            else:
                print("No real-time audio buffer available for comparison")
                
        except Exception as e:
            print(f"Error processing recorded audio: {e}")
            import traceback
            traceback.print_exc()

    def _provide_voice_feedback(self, score, peak_score):
        """Provide user feedback based on voice similarity score"""
        if score >= 85:
            feedback = "Excellent voice consistency!"
            color = 'lime'
        elif score >= 70:
            feedback = "Good voice similarity!"
            color = 'green'
        elif score >= 55:
            feedback = "Fair similarity - try speaking more consistently!"
            color = 'orange'
        elif score >= 40:
            feedback = "Voice patterns differ - practice consistent speech!"
            color = 'yellow'
        else:
            feedback = "Very different voice patterns - keep practicing!"
            color = 'red'
        
        # Update the score display
        score_text = f"Score: {score:.1f}% - {feedback}"
        if hasattr(self, 'score_display'):
            self.score_display.set_text(score_text)
            self.score_display.set_color(color)

    def set_recording_duration(self, duration):
        """Set the recording duration in seconds"""
        self.recording_duration = max(1.0, min(10.0, duration))
        print(f"Recording duration set to {self.recording_duration:.1f} seconds")

    def process_realtime_audio(self):
        """Process real-time audio stream and store SAI frames in buffer"""
        frame_time = self.chunk_size / self.sample_rate
        
        while self.running:
            frame_start = time.time()
            
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Process with CARFAC
                nap_output = self.carfac_realtime.process_chunk(audio_chunk)

                # Process with SAI
                sai_output = self.SAI_realtime.RunSegment(nap_output)
                self.vis_realtime.get_vowel_embedding(nap_output)
                pitch_frame = self.vis_realtime.run_frame(sai_output)

                # Store SAI frame in circular buffer for comparison
                self.realtime_sai_buffer.append(sai_output.copy())
                if len(self.realtime_sai_buffer) > self.max_realtime_buffer_size:
                    self.realtime_sai_buffer.pop(0)

                # Update similarity calculator with real-time SAI frame
                self.similarity_calculator.add_realtime_frame(sai_output)

                # Update visualization
                self.vis_realtime.img[:, :-1] = self.vis_realtime.img[:, 1:]
                self.vis_realtime.draw_column(self.vis_realtime.img[:, -1])

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Real-time audio processing error: {e}")
                continue

    def _setup_visualization(self):
        """Setup visualization with similarity display"""
        self.fig = plt.figure(figsize=(16, 12))
        
        # Create layout with similarity plot and score display
        gs = self.fig.add_gridspec(12, 1, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 0.8, 0.3, 0.3, 0.3])
        
        self.ax_realtime = self.fig.add_subplot(gs[0:8])
        
        # Similarity plot
        self.ax_similarity = self.fig.add_subplot(gs[8])
        self.ax_similarity.set_facecolor('black')
        self.ax_similarity.set_title("Voice Similarity Score", color='white', fontsize=12)
        self.ax_similarity.set_ylim(0, 1)
        self.ax_similarity.set_ylabel("Similarity", color='white')
        self.ax_similarity.tick_params(colors='white')
        
        # Initialize similarity plot line
        self.similarity_line, = self.ax_similarity.plot([], [], 'lime', linewidth=2, label='Current')
        self.similarity_line_smooth, = self.ax_similarity.plot([], [], 'cyan', linewidth=1.5, label='Smoothed')
        self.ax_similarity.legend(loc='upper right')
        self.ax_similarity.grid(True, alpha=0.3)
        
        # Score display area
        self.ax_score = self.fig.add_subplot(gs[9])
        self.ax_score.set_facecolor('black')
        self.ax_score.set_title("Voice Consistency Score", color='white', fontsize=12)
        self.ax_score.set_xlim(0, 1)
        self.ax_score.set_ylim(0, 1)
        self.ax_score.axis('off')
        
        # Score display text
        self.score_display = self.ax_score.text(
            0.5, 0.5, 'Press Record to test voice consistency', 
            transform=self.ax_score.transAxes,
            horizontalalignment='center', verticalalignment='center',
            fontsize=14, color='white', weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8)
        )
        
        # Control buttons area
        self.ax_controls = self.fig.add_subplot(gs[10:12])
        self.ax_controls.axis('off')
        
        # Enhanced colormap
        colors = ['#000022', '#000055', '#0033AA', '#0066FF', '#00AAFF',
                  '#00FFAA', '#33FF77', '#77FF33', '#AAFF00', '#FFAA00',
                  '#FF7700', '#FF3300', '#FF0044', '#CC0077', '#FFFFFF']
        self.cmap = LinearSegmentedColormap.from_list("enhanced_audio", colors, N=256)
        
        # Setup real-time visualization
        self.im_realtime = self.ax_realtime.imshow(
            self.vis_realtime.img, aspect='auto', origin='upper',
            interpolation='bilinear', extent=[0, 200, 0, self.vis_realtime.output.shape[0]]
        )
        self.ax_realtime.set_title("Real-time Microphone SAI", color='white', fontsize=14, pad=20)
        self.ax_realtime.axis('off')
        
        # Add text overlays
        self.transcription_realtime = self.ax_realtime.text(
            0.02, 0.02, '', transform=self.ax_realtime.transAxes,
            verticalalignment='bottom', fontsize=10, color='lime', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        
        # Status indicator for real-time
        self.status_realtime = self.ax_realtime.text(
            0.02, 0.98, '', transform=self.ax_realtime.transAxes,
            verticalalignment='top', fontsize=10, color='white', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        
        # Setup control buttons
        self._setup_control_buttons()
        
        plt.tight_layout()
        self.fig.patch.set_facecolor('black')

    def _setup_control_buttons(self):
        """Setup control buttons"""
        button_height = 0.06
        button_width = 0.08
        button_y = 0.02
        spacing = 0.12
        start_x = 0.2
        
        # Record button
        self.ax_record = plt.axes([start_x, button_y, button_width, button_height], 
                                 facecolor='none', frameon=False)
        self.btn_record = Button(self.ax_record, '🎤 REC', 
                                color=(0, 0.8, 1, 0.8), 
                                hovercolor=(0, 0.8, 1, 1.0))
        self.btn_record.label.set_fontsize(12)
        self.btn_record.label.set_color('white')
        self.btn_record.label.set_weight('bold')
        self.btn_record.on_clicked(self.start_recording)
        
        # Duration down button
        self.ax_duration_down = plt.axes([start_x + spacing, button_y, button_width*0.6, button_height], 
                                        facecolor='none', frameon=False)
        self.btn_duration_down = Button(self.ax_duration_down, '-', 
                                       color=(0.8, 0.8, 0.8, 0.6), 
                                       hovercolor=(0.8, 0.8, 0.8, 0.8))
        self.btn_duration_down.label.set_fontsize(16)
        self.btn_duration_down.label.set_color('white')
        self.btn_duration_down.label.set_weight('bold')
        self.btn_duration_down.on_clicked(lambda x: self.change_recording_duration(-1))
        
        # Duration display
        self.ax_duration_display = plt.axes([start_x + spacing + 0.08, button_y, button_width*0.8, button_height], 
                                           facecolor='none', frameon=False)
        self.duration_text = self.ax_duration_display.text(0.5, 0.5, f'{self.recording_duration:.1f}s', 
                                                          ha='center', va='center', 
                                                          fontsize=12, color='white', weight='bold',
                                                          bbox=dict(boxstyle='round,pad=0.2', 
                                                                   facecolor=(0.2, 0.2, 0.2, 0.8), 
                                                                   edgecolor='white', alpha=0.8))
        self.ax_duration_display.set_xlim(0, 1)
        self.ax_duration_display.set_ylim(0, 1)
        self.ax_duration_display.axis('off')
        
        # Duration up button
        self.ax_duration_up = plt.axes([start_x + spacing + 0.16, button_y, button_width*0.6, button_height], 
                                      facecolor='none', frameon=False)
        self.btn_duration_up = Button(self.ax_duration_up, '+', 
                                     color=(0.8, 0.8, 0.8, 0.6), 
                                     hovercolor=(0.8, 0.8, 0.8, 0.8))
        self.btn_duration_up.label.set_fontsize(16)
        self.btn_duration_up.label.set_color('white')
        self.btn_duration_up.label.set_weight('bold')
        self.btn_duration_up.on_clicked(lambda x: self.change_recording_duration(1))

    def change_recording_duration(self, delta):
        """Change recording duration"""
        new_duration = self.recording_duration + delta
        self.recording_duration = max(1.0, min(10.0, new_duration))
        self.duration_text.set_text(f'{self.recording_duration:.1f}s')

    def update_visualization(self, frame):
        """Update visualization and similarity score"""
        try:
            # Update real-time side
            current_max_rt = np.max(self.vis_realtime.img) if self.vis_realtime.img.size else 1
            self.im_realtime.set_data(self.vis_realtime.img)
            self.im_realtime.set_clim(vmin=0, vmax=max(1, min(255, current_max_rt * 1.3)))
            
            # Calculate and update similarity if we have recorded frames to compare
            if len(self.recorded_sai_frames) > 0 and len(self.realtime_sai_buffer) > 0:
                # Add recorded frames to similarity calculator
                for frame in self.recorded_sai_frames[-3:]:  # Use last few frames
                    self.similarity_calculator.add_recorded_frame(frame)
                
                similarity_score = self.similarity_calculator.calculate_similarity(
                    method=self.similarity_method,
                    temporal_window=3
                )
                
                # Update similarity display history
                if len(self.similarity_display_history) >= self.max_similarity_display_history:
                    self.similarity_display_history.pop(0)
                self.similarity_display_history.append(similarity_score)
                
                # Update similarity plot
                x_data = list(range(len(self.similarity_display_history)))
                
                # Current similarity
                self.similarity_line.set_data(x_data, self.similarity_display_history)
                
                # Smoothed similarity
                smooth_data = [self.similarity_calculator.smoothed_similarity] * len(self.similarity_display_history)
                self.similarity_line_smooth.set_data(x_data, smooth_data)
                
                # Update plot limits
                if x_data:
                    self.ax_similarity.set_xlim(0, max(self.max_similarity_display_history, max(x_data)))
            else:
                # Clear similarity display when no recording
                self.similarity_line.set_data([], [])
                self.similarity_line_smooth.set_data([], [])
            
            # Update text displays
            transcription_rt = self.whisper_realtime.get_display_text()
            if not transcription_rt:
                transcription_rt = "Listening... (speak into microphone)"
            self.transcription_realtime.set_text(transcription_rt)
            
            # Update status for real-time with buffer info
            with self.whisper_buffer_lock_realtime:
                buffer_seconds = len(self.whisper_audio_buffer_realtime) / self.sample_rate
            
            # Add recording status and buffer info to real-time display
            if self.is_recording:
                remaining_time = max(0, self.recording_duration - (time.time() - self.recording_start_time))
                status_info = f"🔴 RECORDING\nTime left: {remaining_time:.1f}s\nSAI Buffer: {len(self.realtime_sai_buffer)} frames"
            else:
                status_info = f"Real-time Status\nAudio Buffer: {buffer_seconds:.1f}s\nSAI Buffer: {len(self.realtime_sai_buffer)} frames"
                if self.score_history:
                    avg_score = np.mean(self.score_history)
                    status_info += f"\nAvg Voice Match: {avg_score:.1f}%"
            
            self.status_realtime.set_text(status_info)
            
            # Update score display for voice comparison
            if self.score_history:
                recent_scores = self.score_history[-3:]  # Last 3 scores
                score_trend = "→"
                if len(recent_scores) >= 2:
                    if recent_scores[-1] > recent_scores[-2]:
                        score_trend = "↗"
                    elif recent_scores[-1] < recent_scores[-2]:
                        score_trend = "↘"
                
                score_summary = f"Voice Similarity - Latest: {self.last_score_percentage:.1f}% {score_trend} | "
                score_summary += f"Best: {max(self.score_history):.1f}% | "
                score_summary += f"Average: {np.mean(self.score_history):.1f}%"
                
                # Color code based on latest score
                if self.last_score_percentage >= 85:
                    color = 'lime'
                elif self.last_score_percentage >= 70:
                    color = 'lightgreen' 
                elif self.last_score_percentage >= 55:
                    color = 'orange'
                elif self.last_score_percentage >= 40:
                    color = 'yellow'
                else:
                    color = 'lightcoral'
                
                if hasattr(self, 'score_display'):
                    self.score_display.set_text(score_summary)
                    self.score_display.set_color(color)
            else:
                if hasattr(self, 'score_display'):
                    self.score_display.set_text('Press Record to compare your voice with real-time audio')
                    self.score_display.set_color('white')
            
        except Exception as e:
            print(f"Visualization update error: {e}")
        
        return [self.im_realtime, self.transcription_realtime, self.status_realtime,
                self.similarity_line, self.similarity_line_smooth, self.score_display]

    def get_pronunciation_statistics(self):
        """Get detailed voice consistency statistics"""
        if not self.score_history:
            return "No voice recordings made yet"
        
        stats = {
            'attempts': len(self.score_history),
            'latest_score': self.last_score_percentage,
            'best_score': max(self.score_history),
            'average_score': np.mean(self.score_history),
            'improvement': 0.0
        }
        
        if len(self.score_history) >= 2:
            stats['improvement'] = self.score_history[-1] - self.score_history[0]
        
        return stats

    def reset_voice_history(self):
        """Reset voice consistency score history"""
        self.score_history.clear()
        self.last_score_percentage = 0.0
        self.recorded_sai_frames.clear()
        if hasattr(self, 'score_display'):
            self.score_display.set_text('Press Record to test voice consistency')
            self.score_display.set_color('white')
        print("Voice consistency history reset")

    def list_audio_devices(self):
        """List available audio input devices"""
        if self.p is None:
            self.p = pyaudio.PyAudio()
        
        print("\nAvailable audio input devices:")
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  Device {i}: {info['name']} (Channels: {info['maxInputChannels']})")
        print()

    def start(self):
        """Start the voice similarity processor"""
        print("Starting Voice-to-Voice Similarity Trainer...")
        
        self.p = pyaudio.PyAudio()
        
        if self.debug:
            self.list_audio_devices()
        
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback,
                start=False
            )
            print(f"Audio stream opened: {self.sample_rate}Hz, {self.chunk_size} frames/buffer")
        except Exception as e:
            print(f"Failed to open audio stream: {e}")
            for i in range(self.p.get_device_count()):
                try:
                    info = self.p.get_device_info_by_index(i)
                    if info['maxInputChannels'] > 0:
                        print(f"Trying device {i}: {info['name']}")
                        self.stream = self.p.open(
                            format=pyaudio.paInt16,
                            channels=1,
                            rate=self.sample_rate,
                            input=True,
                            input_device_index=i,
                            frames_per_buffer=self.chunk_size,
                            stream_callback=self.audio_callback,
                            start=False
                        )
                        print(f"Successfully opened device {i}")
                        break
                except:
                    continue
            else:
                print("Failed to open any audio input device")
                return

        self.running = True
        threading.Thread(target=self.process_realtime_audio, daemon=True).start()
        
        if self.stream:
            self.stream.start_stream()
            print("Real-time audio processing started")
        
        print("\n=== VOICE SIMILARITY TRAINER ===")
        print("1. Real-time microphone input is continuously processed")
        print("2. Click 🎤 REC button to record your voice")
        print("3. Your recorded voice will be compared against recent real-time audio")
        print("4. Get a similarity percentage score showing voice consistency")
        print("5. Use +/- buttons to adjust recording duration")
        print("Press Ctrl+C to stop.")
        
        real_time_interval = (self.chunk_size / self.sample_rate) * 1000
        animation_interval = max(10, int(real_time_interval))
        
        print(f"Animation interval: {animation_interval}ms")
        
        self.animation = animation.FuncAnimation(
            self.fig, self.update_visualization, interval=animation_interval, 
            blit=False, cache_frame_data=False
        )
        plt.show()

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.is_recording = False
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
        except:
            pass
        try:
            if self.p:
                self.p.terminate()
        except:
            pass

    def stop(self):
        """Stop the system"""
        self.cleanup()
        plt.close('all')
        print("Voice similarity trainer stopped.")
        
        # Print final statistics
        if self.score_history:
            stats = self.get_pronunciation_statistics()
            print(f"\n=== FINAL STATISTICS ===")
            print(f"Total attempts: {stats['attempts']}")
            print(f"Best score: {stats['best_score']:.1f}%")
            print(f"Average score: {stats['average_score']:.1f}%")
            print(f"Latest score: {stats['latest_score']:.1f}%")
            if stats['improvement'] > 0:
                print(f"Improvement: +{stats['improvement']:.1f}% (Great progress!)")
            elif stats['improvement'] < 0:
                print(f"Score change: {stats['improvement']:.1f}% (Keep practicing!)")
            else:
                print("Keep practicing to improve voice consistency!")

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description='Voice Similarity Trainer - Compare recorded voice vs real-time or file')
    parser.add_argument('--audio-file', help='Path to reference audio file (optional)')
    parser.add_argument('--chunk-size', type=int, default=512, help='Audio chunk size (default: 512)')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate (default: 16000)')
    parser.add_argument('--sai-width', type=int, default=400, help='SAI width (default: 400)')
    parser.add_argument('--whisper-model', default='tiny', help='Whisper model (default: tiny)')
    parser.add_argument('--whisper-interval', type=float, default=2.0, help='Whisper processing interval in seconds (default: 2.0)')
    parser.add_argument('--similarity-method', default='cosine', choices=['cosine', 'correlation', 'euclidean', 'spectral'], 
                        help='Similarity calculation method (default: cosine)')
    parser.add_argument('--recording-duration', type=float, default=5.0, help='Default recording duration in seconds (default: 5.0)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--mode', default='voice-to-voice', choices=['voice-to-voice', 'voice-to-file'], 
                        help='Comparison mode: voice-to-voice or voice-to-file (default: voice-to-voice)')
    
    args = parser.parse_args()
    
    # Check if audio file exists when specified
    if args.audio_file and not os.path.exists(args.audio_file):
        print(f"Warning: Audio file '{args.audio_file}' not found")
        print("Switching to voice-to-voice mode...")
        args.audio_file = None
        args.mode = 'voice-to-voice'
    
    try:
        if args.mode == 'voice-to-file' or args.audio_file:
            print(f"Starting Voice-to-File comparison mode with reference file: {args.audio_file}")
            # Use original dual processor for voice-to-file comparison
            from your_original_dual_processor import DualSAIProcessor  # You'll need to import this
            processor = DualSAIProcessor(
                audio_file_path=args.audio_file,
                chunk_size=args.chunk_size,
                sample_rate=args.sample_rate,
                sai_width=args.sai_width,
                whisper_model=args.whisper_model,
                whisper_interval=args.whisper_interval,
                debug=args.debug,
                similarity_method=args.similarity_method
            )
        else:
            print("Starting Voice-to-Voice comparison mode")
            processor = VoiceSimilarityProcessor(
                chunk_size=args.chunk_size,
                sample_rate=args.sample_rate,
                sai_width=args.sai_width,
                whisper_model=args.whisper_model,
                whisper_interval=args.whisper_interval,
                debug=args.debug,
                similarity_method=args.similarity_method
            )
        
        # Set custom recording duration if specified
        processor.set_recording_duration(args.recording_duration)
        
        processor.start()
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error in similarity trainer: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        processor = VoiceSimilarityProcessor(debug=True)
        try:
            processor.start()
        except KeyboardInterrupt:
            print("\nShutting down...")
            processor.stop()
    else:
        main()