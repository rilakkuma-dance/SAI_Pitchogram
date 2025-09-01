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

# ---------------- Dual SAI Processor ----------------
class DualSAIProcessor:
    def __init__(self, audio_file_path=None, chunk_size=1024, sample_rate=16000, sai_width=200,
                 whisper_model="tiny", whisper_interval=1.5, 
                 enable_translation=False, from_lang="zh", to_lang="en",
                 debug=True, playback_speed=3.0, loop_audio=True, enable_caching=True, 
                 enable_audio_playback=False):
        
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sai_width = sai_width
        self.whisper_interval = whisper_interval
        self.debug = debug
        self.playback_speed = playback_speed
        self.loop_audio = loop_audio  # NEW: Enable/disable looping
        self.enable_caching = enable_caching  # NEW: Enable/disable SAI caching
        self.enable_audio_playback = enable_audio_playback  # NEW: Enable/disable audio playback
        
        # Audio playback setup
        self.playback_stream = None
        self.playback_position = 0
        self.last_playback_time = time.time()
        self.is_playing = False
        self.playback_paused = False
        
        # Initialize processing components for both sides
        self.carfac_realtime = RealCARFACProcessor(fs=sample_rate)
        self.carfac_file = RealCARFACProcessor(fs=sample_rate)
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
        
        # Create separate SAI instances
        self.SAI_realtime = sai.SAI(self.sai_params)
        self.SAI_file = sai.SAI(self.sai_params)

        # Visualization handlers for both sides
        self.vis_realtime = VisualizationHandler(sample_rate, sai_params=self.sai_params)
        self.vis_file = VisualizationHandler(sample_rate, sai_params=self.sai_params)

        # Whisper handlers
        self.whisper_realtime = WhisperHandler(model_name=whisper_model, debug=debug)
        self.whisper_file = WhisperHandler(model_name=whisper_model, debug=debug)

        # Real-time audio setup
        self.audio_queue = queue.Queue(maxsize=50)
        self.whisper_audio_buffer_realtime = []
        self.whisper_buffer_lock_realtime = threading.Lock()
        self.last_whisper_time_realtime = time.time()
        
        # File processing setup
        self.audio_file_path = audio_file_path
        self.audio_data = None
        self.original_sr = None
        self.current_position = 0
        self.chunks_processed = 0
        self.duration = 0
        self.total_samples = 0
        self.loop_count = 0  # NEW: Track how many times we've looped
        
        # NEW: SAI Caching System
        self.sai_cache = {}  # Cache for SAI results: chunk_index -> (nap_output, sai_output, vis_output)
        self.cache_state_snapshots = {}  # Store CARFAC/SAI states at key points
        self.is_precalculating = False
        self.precalculation_progress = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        if audio_file_path and os.path.exists(audio_file_path):
            self._load_audio_file()
            if self.enable_caching and self.audio_data is not None:
                print("Precalculating SAI for optimal looping performance...")
                self._precalculate_sai()
        
        self.whisper_audio_buffer_file = []
        self.whisper_buffer_lock_file = threading.Lock()
        self.last_whisper_time_file = 0

        # Audio stream
        self.p = None
        self.stream = None
        self.running = False
        
        # Setup visualization
        self._setup_dual_visualization()

    def setup_audio_playback(self):
        """Setup audio playback stream for file audio"""
        if not self.enable_audio_playback or self.audio_data is None:
            return
        
        try:
            # Convert audio to int16 for playback
            playback_audio = (self.audio_data * 32767).astype(np.int16)
            
            def playback_callback(in_data, frame_count, time_info, status):
                try:
                    if not self.is_playing or self.playback_paused:
                        return (np.zeros(frame_count, dtype=np.int16).tobytes(), pyaudio.paContinue)
                    
                    if self.playback_position >= len(playback_audio):
                        if self.loop_audio:
                            self.playback_position = 0
                        else:
                            self.is_playing = False
                            self.btn_play.label.set_text('▶ Play')
                            self.btn_play.color = 'lightgreen'
                            return (np.zeros(frame_count, dtype=np.int16).tobytes(), pyaudio.paComplete)
                    
                    end_pos = min(self.playback_position + frame_count, len(playback_audio))
                    chunk = playback_audio[self.playback_position:end_pos]
                    
                    if len(chunk) < frame_count:
                        if self.loop_audio and len(playback_audio) > 0:
                            # Fill remainder with beginning of file
                            remaining = frame_count - len(chunk)
                            loop_chunk = playback_audio[:min(remaining, len(playback_audio))]
                            chunk = np.concatenate([chunk, loop_chunk])
                            self.playback_position = len(loop_chunk)
                        else:
                            # Pad with silence
                            chunk = np.pad(chunk, (0, frame_count - len(chunk)), 'constant')
                            self.playback_position = end_pos
                    else:
                        self.playback_position = end_pos
                    
                    # Apply playback speed (simple resampling)
                    if self.playback_speed != 1.0:
                        # This is a simple approach - for better quality, use librosa
                        indices = np.arange(0, len(chunk), self.playback_speed).astype(int)
                        indices = indices[indices < len(chunk)]
                        if len(indices) > 0:
                            chunk = chunk[indices]
                        
                        # Pad or truncate to frame_count
                        if len(chunk) < frame_count:
                            chunk = np.pad(chunk, (0, frame_count - len(chunk)), 'constant')
                        else:
                            chunk = chunk[:frame_count]
                    
                    return (chunk.tobytes(), pyaudio.paContinue)
                    
                except Exception as e:
                    print(f"Playback callback error: {e}")
                    return (np.zeros(frame_count, dtype=np.int16).tobytes(), pyaudio.paComplete)
            
            self.playback_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=playback_callback,
                start=False
            )
            
            print(f"Audio playback setup complete at {self.playback_speed}x speed")
            
        except Exception as e:
            print(f"Failed to setup audio playback: {e}")
            self.enable_audio_playback = False

    def _load_audio_file(self):
        """Load the audio file for file processing"""
        print(f"Loading audio file: {self.audio_file_path}")
        self.audio_data, self.original_sr = librosa.load(self.audio_file_path, sr=None)
        
        if self.original_sr != self.sample_rate:
            print(f"Resampling from {self.original_sr}Hz to {self.sample_rate}Hz")
            self.audio_data = librosa.resample(self.audio_data, orig_sr=self.original_sr, target_sr=self.sample_rate)
        
        if np.max(np.abs(self.audio_data)) > 0:
            self.audio_data = self.audio_data / np.max(np.abs(self.audio_data)) * 0.9
        
        self.total_samples = len(self.audio_data)
        self.duration = self.total_samples / self.sample_rate
        print(f"Audio loaded: {self.duration:.2f} seconds, {self.total_samples} samples")
        
        if self.loop_audio:
            print("Audio looping is enabled - file will repeat continuously")
        
        if self.enable_caching:
            print("SAI caching is enabled for optimal performance")

    def _precalculate_sai(self):
        """Precalculate SAI for the entire audio file"""
        if self.audio_data is None:
            return
        
        print(f"Precalculating SAI for {self.duration:.2f}s audio file...")
        self.is_precalculating = True
        
        # Create temporary CARFAC and SAI instances for precalculation
        temp_carfac = RealCARFACProcessor(fs=self.sample_rate)
        temp_sai = sai.SAI(self.sai_params)
        temp_vis = VisualizationHandler(self.sample_rate, sai_params=self.sai_params)
        
        # Store initial states
        self.cache_state_snapshots[0] = {
            'carfac_state': temp_carfac.state,
            'sai_state': temp_sai,  # SAI doesn't expose internal state easily
            'vis_cgram': temp_vis.cgram.copy()
        }
        
        temp_position = 0
        chunk_index = 0
        total_chunks = int(np.ceil(self.total_samples / self.chunk_size))
        
        start_time = time.time()
        
        while temp_position < self.total_samples:
            try:
                # Get chunk
                end_position = min(temp_position + self.chunk_size, self.total_samples)
                chunk = self.audio_data[temp_position:end_position]
                
                if len(chunk) < self.chunk_size:
                    chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), 'constant')
                
                chunk = chunk.astype(np.float32)
                
                # Validate chunk
                if np.any(np.isnan(chunk)) or np.any(np.isinf(chunk)):
                    print(f"Warning: Invalid audio data at chunk {chunk_index}")
                    chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Process chunk
                nap_output = temp_carfac.process_chunk(chunk)
                sai_output = temp_sai.RunSegment(nap_output)
                temp_vis.get_vowel_embedding(nap_output)
                vis_output = temp_vis.run_frame(sai_output)
                
                # Validate outputs
                if nap_output is None or sai_output is None or vis_output is None:
                    print(f"Warning: Null output at chunk {chunk_index}")
                    continue
                
                # Cache results
                self.sai_cache[chunk_index] = {
                    'nap_output': nap_output.copy(),
                    'sai_output': sai_output.copy(),
                    'vis_output': vis_output.copy(),
                    'vowel_coords': temp_vis.vowel_coords.copy(),
                    'cgram': temp_vis.cgram.copy()
                }
                
                # Store state snapshots every 100 chunks for state restoration
                if chunk_index % 100 == 0:
                    self.cache_state_snapshots[chunk_index] = {
                        'carfac_state': temp_carfac.state,
                        'vis_cgram': temp_vis.cgram.copy()
                    }
                
                temp_position = end_position
                chunk_index += 1
                self.precalculation_progress = (chunk_index / total_chunks) * 100
                
                # Progress update
                if chunk_index % 50 == 0 or chunk_index == total_chunks:
                    elapsed = time.time() - start_time
                    eta = (elapsed / chunk_index) * (total_chunks - chunk_index) if chunk_index > 0 else 0
                    print(f"  Progress: {chunk_index}/{total_chunks} chunks ({self.precalculation_progress:.1f}%) - ETA: {eta:.1f}s")
                    
            except Exception as e:
                print(f"Error precalculating chunk {chunk_index}: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                temp_position = end_position
                chunk_index += 1
                continue
        
        elapsed_time = time.time() - start_time
        successful_chunks = len(self.sai_cache)
        print(f"✓ SAI precalculation complete: {successful_chunks}/{total_chunks} chunks in {elapsed_time:.2f}s")
        if successful_chunks < total_chunks:
            print(f"  Warning: {total_chunks - successful_chunks} chunks failed to process")
        print(f"  Cache size: {len(self.sai_cache)} entries")
        print(f"  Memory usage: ~{len(self.sai_cache) * self.chunk_size * 4 / 1024 / 1024:.1f} MB")
        
        self.is_precalculating = False

    def get_cached_sai_result(self, chunk_index):
        """Get cached SAI result for a given chunk index"""
        if not self.enable_caching or chunk_index not in self.sai_cache:
            self.cache_misses += 1
            return None
        
        self.cache_hits += 1
        return self.sai_cache[chunk_index]

    def process_file_chunk_optimized(self, chunk, chunk_index):
        """Process file chunk with caching optimization"""
        cached_result = self.get_cached_sai_result(chunk_index)
        
        if cached_result is not None:
            # Use cached results
            nap_output = cached_result['nap_output']
            sai_output = cached_result['sai_output']
            vis_output = cached_result['vis_output']
            
            # Update visualization state
            self.vis_file.vowel_coords = cached_result['vowel_coords'].copy()
            self.vis_file.cgram = cached_result['cgram'].copy()
            
            # IMPORTANT: Still need to update the visualization image!
            self.vis_file.output = vis_output.copy()
            
            return nap_output, sai_output, vis_output
        else:
            # Calculate normally (fallback for when caching is disabled or cache miss)
            nap_output = self.carfac_file.process_chunk(chunk)
            sai_output = self.SAI_file.RunSegment(nap_output)
            self.vis_file.get_vowel_embedding(nap_output)
            vis_output = self.vis_file.run_frame(sai_output)
            
            # Cache the result if caching is enabled
            if self.enable_caching:
                self.sai_cache[chunk_index] = {
                    'nap_output': nap_output.copy(),
                    'sai_output': sai_output.copy(),
                    'vis_output': vis_output.copy(),
                    'vowel_coords': self.vis_file.vowel_coords.copy(),
                    'cgram': self.vis_file.cgram.copy()
                }
            
            return nap_output, sai_output, vis_output

    def restore_state_from_cache(self, target_chunk_index):
        """Restore CARFAC/SAI state from nearest cached snapshot"""
        if not self.enable_caching:
            return
        
        # Find nearest snapshot
        snapshot_indices = [idx for idx in self.cache_state_snapshots.keys() if idx <= target_chunk_index]
        if not snapshot_indices:
            return
        
        nearest_index = max(snapshot_indices)
        snapshot = self.cache_state_snapshots[nearest_index]
        
        # Restore visualization state
        if 'vis_cgram' in snapshot:
            self.vis_file.cgram = snapshot['vis_cgram'].copy()

    def get_current_chunk_index(self):
        """Get the current chunk index based on position"""
        return int(self.current_position / self.chunk_size)

    def _setup_dual_visualization(self):
        """Setup dual screen visualization with interactive controls"""
        self.fig = plt.figure(figsize=(20, 12))  # Increased height for buttons
        
        # Create two subplots side by side with space for buttons at bottom
        gs = self.fig.add_gridspec(12, 2, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.3, 0.3])
        
        self.ax_realtime = self.fig.add_subplot(gs[0:10, 0])
        self.ax_file = self.fig.add_subplot(gs[0:10, 1])
        
        # Control buttons area (spans both columns)
        self.ax_controls = self.fig.add_subplot(gs[10:12, :])
        self.ax_controls.axis('off')
        
        # Enhanced colormap
        colors = ['#000022', '#000055', '#0033AA', '#0066FF', '#00AAFF',
                  '#00FFAA', '#33FF77', '#77FF33', '#AAFF00', '#FFAA00',
                  '#FF7700', '#FF3300', '#FF0044', '#CC0077', '#FFFFFF']
        self.cmap = LinearSegmentedColormap.from_list("enhanced_audio", colors, N=256)
        
        # Setup left side (real-time)
        self.im_realtime = self.ax_realtime.imshow(
            self.vis_realtime.img, aspect='auto', origin='upper',
            interpolation='bilinear', extent=[0, 200, 0, self.vis_realtime.output.shape[0]]
        )
        self.ax_realtime.set_title("Real-time Microphone SAI", color='white', fontsize=14, pad=20)
        self.ax_realtime.axis('off')
        
        # Setup right side (file)
        self.im_file = self.ax_file.imshow(
            self.vis_file.img, aspect='auto', origin='upper',
            interpolation='bilinear', extent=[0, 200, 0, self.vis_file.output.shape[0]]
        )
        file_title = f"File SAI: {os.path.basename(self.audio_file_path) if self.audio_file_path else 'No file loaded'}"
        if self.loop_audio:
            file_title += " (LOOPING)"
        self.ax_file.set_title(file_title, color='white', fontsize=14, pad=20)
        self.ax_file.axis('off')
        
        # Add text overlays for each side
        self.transcription_realtime = self.ax_realtime.text(
            0.02, 0.02, '', transform=self.ax_realtime.transAxes,
            verticalalignment='bottom', fontsize=10, color='lime', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        
        self.transcription_file = self.ax_file.text(
            0.02, 0.02, '', transform=self.ax_file.transAxes,
            verticalalignment='bottom', fontsize=10, color='cyan', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        
        # Progress indicator for file
        self.progress_file = self.ax_file.text(
            0.02, 0.98, '', transform=self.ax_file.transAxes,
            verticalalignment='top', fontsize=10, color='yellow', weight='bold',
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
        """Setup modern, attractive control buttons"""
        button_height = 0.04
        button_width = 0.06
        button_y = 0.01
        spacing = 0.08
        start_x = 0.25
        
        # Play/Pause button
        self.ax_play = plt.axes([start_x, button_y, button_width, button_height], 
                               facecolor='none', frameon=False)
        self.btn_play = Button(self.ax_play, '▶', 
                              color=(0, 1, 0.4, 0.8), 
                              hovercolor=(0, 1, 0.4, 1.0))
        self.btn_play.label.set_fontsize(16)
        self.btn_play.label.set_color('white')
        self.btn_play.label.set_weight('bold')
        self.btn_play.on_clicked(self.toggle_playback)
        
        # Stop button
        self.ax_stop = plt.axes([start_x + spacing, button_y, button_width, button_height], 
                               facecolor='none', frameon=False)
        self.btn_stop = Button(self.ax_stop, '⏹', 
                              color=(1, 0.4, 0.4, 0.8), 
                              hovercolor=(1, 0.4, 0.4, 1.0))
        self.btn_stop.label.set_fontsize(16)
        self.btn_stop.label.set_color('white')
        self.btn_stop.label.set_weight('bold')
        self.btn_stop.on_clicked(self.stop_playback)
        
        # Speed down button
        self.ax_speed_down = plt.axes([start_x + spacing*2, button_y, button_width, button_height], 
                                     facecolor='none', frameon=False)
        self.btn_speed_down = Button(self.ax_speed_down, '◀◀', 
                                    color=(0.4, 0.6, 1, 0.8), 
                                    hovercolor=(0.4, 0.6, 1, 1.0))
        self.btn_speed_down.label.set_fontsize(12)
        self.btn_speed_down.label.set_color('white')
        self.btn_speed_down.label.set_weight('bold')
        self.btn_speed_down.on_clicked(self.decrease_speed)
        
        # Speed indicator (non-clickable)
        self.ax_speed_display = plt.axes([start_x + spacing*3, button_y, button_width, button_height], 
                                        facecolor='none', frameon=False)
        self.speed_text = self.ax_speed_display.text(0.5, 0.5, f'{self.playback_speed:.1f}x', 
                                                    ha='center', va='center', 
                                                    fontsize=12, color='white', weight='bold',
                                                    bbox=dict(boxstyle='round,pad=0.2', 
                                                             facecolor=(1, 1, 1, 0.1), 
                                                             edgecolor='white', alpha=0.8))
        self.ax_speed_display.set_xlim(0, 1)
        self.ax_speed_display.set_ylim(0, 1)
        self.ax_speed_display.axis('off')
        
        # Speed up button
        self.ax_speed_up = plt.axes([start_x + spacing*4, button_y, button_width, button_height], 
                                   facecolor='none', frameon=False)
        self.btn_speed_up = Button(self.ax_speed_up, '▶▶', 
                                  color=(0.4, 0.6, 1, 0.8), 
                                  hovercolor=(0.4, 0.6, 1, 1.0))
        self.btn_speed_up.label.set_fontsize(12)
        self.btn_speed_up.label.set_color('white')
        self.btn_speed_up.label.set_weight('bold')
        self.btn_speed_up.on_clicked(self.increase_speed)
        
        # Loop toggle button
        self.ax_loop = plt.axes([start_x + spacing*5, button_y, button_width, button_height], 
                               facecolor='none', frameon=False)
        loop_symbol = '🔄' if self.loop_audio else '↗'
        loop_color = (1, 0.8, 0, 0.8) if self.loop_audio else (0.5, 0.5, 0.5, 0.6)
        hover_color = (1, 0.8, 0, 1.0) if self.loop_audio else (0.5, 0.5, 0.5, 0.8)
        self.btn_loop = Button(self.ax_loop, loop_symbol, 
                              color=loop_color, 
                              hovercolor=hover_color)
        self.btn_loop.label.set_fontsize(14)
        self.btn_loop.label.set_color('white')
        self.btn_loop.on_clicked(self.toggle_loop)

    def toggle_playback(self, event):
        """Toggle audio playback on/off"""
        if self.audio_data is None:
            print("No audio file loaded")
            return
        
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self):
        """Start audio playback"""
        if self.audio_data is None:
            print("No audio file loaded")
            return
            
        try:
            if self.playback_stream is None:
                self.setup_audio_playback()
            
            if self.playback_stream and not self.is_playing:
                if not self.playback_stream.is_active():
                    self.playback_stream.start_stream()
                self.is_playing = True
                self.playback_paused = False
                self.btn_play.label.set_text('⏸')
                self.btn_play.color = (1, 0.8, 0, 0.8)  # Orange when playing
                print("Audio playback started")
            
        except Exception as e:
            print(f"Failed to start playback: {e}")

    def pause_playback(self):
        """Pause audio playback"""
        if self.playback_stream and self.is_playing:
            self.playback_stream.stop_stream()
            self.is_playing = False
            self.playback_paused = True
            self.btn_play.label.set_text('▶')
            self.btn_play.color = (0, 1, 0.4, 0.8)  # Green when paused
            print("Audio playback paused")

    def stop_playback(self, event):
        """Stop audio playback and reset position"""
        if self.playback_stream:
            if self.is_playing:
                self.playback_stream.stop_stream()
            self.is_playing = False
            self.playback_paused = False
            self.playback_position = 0
            self.btn_play.label.set_text('▶')
            self.btn_play.color = (0, 1, 0.4, 0.8)  # Green when stopped
            print("Audio playback stopped and reset")

    def increase_speed(self, event):
        """Increase playback speed"""
        old_speed = self.playback_speed
        self.playback_speed = min(5.0, self.playback_speed + 0.5)
        if self.playback_speed != old_speed:
            self.speed_text.set_text(f'{self.playback_speed:.1f}x')
            print(f"Speed increased to {self.playback_speed:.1f}x")
            if self.is_playing:
                # Restart playback with new speed
                self.pause_playback()
                time.sleep(0.1)
                self.start_playback()

    def decrease_speed(self, event):
        """Decrease playback speed"""
        old_speed = self.playback_speed
        self.playback_speed = max(0.25, self.playback_speed - 0.5)
        if self.playback_speed != old_speed:
            self.speed_text.set_text(f'{self.playback_speed:.1f}x')
            print(f"Speed decreased to {self.playback_speed:.1f}x")
            if self.is_playing:
                # Restart playback with new speed
                self.pause_playback()
                time.sleep(0.1)
                self.start_playback()

    def toggle_loop(self, event):
        """Toggle loop mode"""
        self.loop_audio = not self.loop_audio
        
        # Update button appearance
        loop_symbol = '🔄' if self.loop_audio else '↗'
        loop_color = (1, 0.8, 0, 0.8) if self.loop_audio else (0.5, 0.5, 0.5, 0.6)
        self.btn_loop.label.set_text(loop_symbol)
        self.btn_loop.color = loop_color
        
        # Update file title
        file_title = f"File SAI: {os.path.basename(self.audio_file_path) if self.audio_file_path else 'No file loaded'}"
        if self.loop_audio:
            file_title += " (LOOPING)"
        self.ax_file.set_title(file_title, color='white', fontsize=14, pad=20)
        
        print(f"Loop mode: {'ON' if self.loop_audio else 'OFF'}")

    def get_next_file_chunk(self):
        """Get next chunk from file with looping support and chunk tracking"""
        if self.audio_data is None:
            return None, -1
        
        # If we've reached the end and looping is enabled, reset position
        if self.current_position >= self.total_samples:
            if self.loop_audio:
                self.current_position = 0
                self.loop_count += 1
                if self.debug:
                    print(f"Audio file looped (loop #{self.loop_count})")
                    if self.enable_caching:
                        cache_efficiency = (self.cache_hits / max(1, self.cache_hits + self.cache_misses)) * 100
                        print(f"  Cache efficiency: {cache_efficiency:.1f}% ({self.cache_hits} hits, {self.cache_misses} misses)")
                
                # Reset some states but maintain cache
                # self.SAI_file = sai.SAI(self.sai_params)  # Commented out for smoother transitions
            else:
                return None, -1
        
        chunk_index = self.get_current_chunk_index()
        end_position = min(self.current_position + self.chunk_size, self.total_samples)
        
        # Handle the case where we might be at the exact end
        if self.current_position >= self.total_samples:
            if self.loop_audio:
                self.current_position = 0
                chunk_index = 0
                end_position = min(self.chunk_size, self.total_samples)
            else:
                return None, -1
        
        chunk = self.audio_data[self.current_position:end_position]
        
        # Handle partial chunks at the end of the file
        if len(chunk) < self.chunk_size:
            if self.loop_audio and self.current_position + len(chunk) >= self.total_samples:
                # We're at the end and looping is enabled
                # Fill the remaining space with the beginning of the file
                remaining_samples = self.chunk_size - len(chunk)
                if remaining_samples > 0 and self.total_samples > 0:
                    loop_start = self.audio_data[:min(remaining_samples, self.total_samples)]
                    chunk = np.concatenate([chunk, loop_start])
                    # Update position for next iteration
                    self.current_position = len(loop_start)
                else:
                    self.current_position = end_position
            else:
                # Pad with zeros if not looping or if there's not enough data
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), 'constant')
                self.current_position = end_position
        else:
            self.current_position = end_position
        
        # Validate chunk
        if len(chunk) != self.chunk_size:
            print(f"Warning: Chunk {chunk_index} has incorrect size: {len(chunk)} vs {self.chunk_size}")
            chunk = np.pad(chunk, (0, max(0, self.chunk_size - len(chunk))), 'constant')
            chunk = chunk[:self.chunk_size]  # Truncate if too long
        
        return chunk.astype(np.float32), chunk_index

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Real-time audio callback"""
        try:
            audio_float = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            
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

    def process_realtime_audio(self):
        """Process real-time audio stream with controlled rate"""
        frame_time = self.chunk_size / self.sample_rate  # Time per chunk in seconds
        
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

                # Update visualization
                self.vis_realtime.img[:, :-1] = self.vis_realtime.img[:, 1:]
                self.vis_realtime.draw_column(self.vis_realtime.img[:, -1])

                # Optional: Add frame rate limiting to match file side better
                # frame_end = time.time()
                # frame_duration = frame_end - frame_start
                # if frame_duration < frame_time:
                #     time.sleep(frame_time - frame_duration)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Real-time audio processing error: {e}")
                continue

    def process_file_chunk_for_whisper(self, chunk, timestamp):
        """Process file chunk for Whisper"""
        with self.whisper_buffer_lock_file:
            self.whisper_audio_buffer_file.extend(chunk)
            
            if (timestamp - self.last_whisper_time_file) >= self.whisper_interval:
                if len(self.whisper_audio_buffer_file) >= int(self.sample_rate * 0.8):
                    audio_to_process = np.array(self.whisper_audio_buffer_file, dtype=np.float32)
                    
                    overlap_size = int(self.sample_rate * 0.5)
                    if len(self.whisper_audio_buffer_file) > overlap_size:
                        self.whisper_audio_buffer_file = self.whisper_audio_buffer_file[-overlap_size:]
                    else:
                        self.whisper_audio_buffer_file = []
                    
                    threading.Thread(
                        target=self._process_whisper_file_chunk,
                        args=(audio_to_process, timestamp),
                        daemon=True
                    ).start()
                    
                    self.last_whisper_time_file = timestamp

    def _process_whisper_file_chunk(self, audio_data, timestamp):
        """Process Whisper transcription for file"""
        try:
            text = self.whisper_file.transcribe_audio(audio_data, language='en')
            if text and len(text.strip()) > 0:
                # Include loop information in the timestamp
                loop_info = f" (Loop #{self.loop_count})" if self.loop_count > 0 and self.loop_audio else ""
                timestamped_text = f"[{timestamp:.1f}s{loop_info}] {text}"
                self.whisper_file.add_transcription_line(timestamped_text)
        except Exception as e:
            print(f"File Whisper processing error: {e}")
            
    def update_visualization(self, frame):
        """Update both visualizations"""
        try:
            # Update real-time side
            current_max_rt = np.max(self.vis_realtime.img) if self.vis_realtime.img.size else 1
            self.im_realtime.set_data(self.vis_realtime.img)
            self.im_realtime.set_clim(vmin=0, vmax=max(1, min(255, current_max_rt * 1.3)))
            
            # Update file side - Process multiple chunks if needed to maintain speed
            if self.audio_data is not None:
                # Calculate how many chunks we should process this frame to maintain real-time speed
                target_chunks_per_frame = max(1, int(self.playback_speed))
                
                for _ in range(target_chunks_per_frame):
                    # Process file chunk
                    chunk, chunk_index = self.get_next_file_chunk()
                    if chunk is not None and chunk_index >= 0:
                        try:
                            # Use optimized processing with caching
                            nap_output, sai_output, vis_output = self.process_file_chunk_optimized(chunk, chunk_index)

                            # Update visualization - CRITICAL: This was missing proper updates!
                            self.vis_file.img[:, :-1] = self.vis_file.img[:, 1:]
                            self.vis_file.draw_column(self.vis_file.img[:, -1])

                            # Calculate current time (accounting for loops)
                            current_file_position = self.current_position
                            if self.current_position == 0 and self.chunks_processed > 0:
                                # We just looped, use the end of file time
                                current_time = self.duration
                            else:
                                current_time = (current_file_position / self.total_samples) * self.duration if self.total_samples > 0 else 0
                            
                            # Process for Whisper (only once per animation frame to avoid overwhelming)
                            if _ == 0:  # Only process Whisper on first chunk of the frame
                                self.process_file_chunk_for_whisper(chunk, current_time)
                            
                            self.chunks_processed += 1
                            
                        except Exception as e:
                            print(f"Error processing file chunk {chunk_index}: {e}")
                            if self.debug:
                                import traceback
                                traceback.print_exc()
                            break
                    else:
                        break  # No more chunks available

            current_max_file = np.max(self.vis_file.img) if self.vis_file.img.size else 1
            self.im_file.set_data(self.vis_file.img)
            self.im_file.set_clim(vmin=0, vmax=max(1, min(255, current_max_file * 1.3)))
            
            # Update text displays
            transcription_rt = self.whisper_realtime.get_display_text()
            if not transcription_rt:
                transcription_rt = "Listening... (speak into microphone)"
            self.transcription_realtime.set_text(transcription_rt)
            
            transcription_file = self.whisper_file.get_display_text()
            if not transcription_file:
                if self.audio_data is not None:
                    transcription_file = "Processing file..." if self.chunks_processed == 0 else "File processing..."
                else:
                    transcription_file = "No audio file loaded"
            self.transcription_file.set_text(transcription_file)
            
            # Update progress for file (with loop information)
            if self.audio_data is not None:
                # Calculate progress within current loop
                current_file_position = self.current_position
                if current_file_position == 0 and self.chunks_processed > 0:
                    current_file_position = self.total_samples
                
                current_time_in_loop = (current_file_position / self.total_samples) * self.duration if self.total_samples > 0 else 0
                progress_percent = (current_file_position / self.total_samples) * 100 if self.total_samples > 0 else 0
                
                # Total elapsed time across all loops
                total_elapsed = (self.loop_count * self.duration) + current_time_in_loop
                
                progress_info = f"File: {current_time_in_loop:.1f}s / {self.duration:.1f}s ({progress_percent:.1f}%)"
                if self.loop_audio:
                    progress_info += f"\nLoop #{self.loop_count + 1} | Total: {total_elapsed:.1f}s"
                progress_info += f"\nSpeed: {self.playback_speed:.1f}x"
                
                # Add audio playback status
                if self.audio_data is not None:
                    playback_status = "🔊 Playing" if self.is_playing else ("⏸ Paused" if self.playback_paused else "⏹ Stopped")
                    progress_info += f" | Audio: {playback_status}"
                
                # Add caching info if enabled
                if self.enable_caching and (self.cache_hits + self.cache_misses) > 0:
                    cache_efficiency = (self.cache_hits / (self.cache_hits + self.cache_misses)) * 100
                    progress_info += f"\nCache: {cache_efficiency:.0f}% hit rate"
                
                self.progress_file.set_text(progress_info)
            else:
                self.progress_file.set_text("No file loaded")
                
            # Update status for real-time
            with self.whisper_buffer_lock_realtime:
                buffer_seconds = len(self.whisper_audio_buffer_realtime) / self.sample_rate
            status_info = f"Real-time Status\nBuffer: {buffer_seconds:.1f}s"
            self.status_realtime.set_text(status_info)
            
        except Exception as e:
            print(f"Visualization update error: {e}")
        
        return [self.im_realtime, self.im_file, self.transcription_realtime, 
                self.transcription_file, self.progress_file, self.status_realtime]

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
        """Start the dual SAI processor"""
        print("Starting Dual SAI Visualization System...")
        
        # Initialize PyAudio for real-time processing
        self.p = pyaudio.PyAudio()
        
        if self.debug:
            self.list_audio_devices()
        
        try:
            # Setup audio stream
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
            # Try to find a working input device
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
                print("Failed to open any audio input device - continuing with file only")

        # Setup audio playback if enabled
        if self.enable_audio_playback:
            self.setup_audio_playback()
            # Auto-start playback if enabled
            self.start_playback()

        # Start processing threads
        self.running = True
        threading.Thread(target=self.process_realtime_audio, daemon=True).start()
        
        # Start audio streams
        if self.stream:
            self.stream.start_stream()
            print("Real-time audio processing started")
        
        if self.playback_stream and self.enable_audio_playback:
            print("Audio playback ready (use controls to start)")
        elif self.audio_data is not None:
            print("Audio playback available (use Play button)")
            
        # Print status
        if self.audio_data is not None:
            loop_status = "with looping enabled" if self.loop_audio else "single playthrough"
            cache_status = "with SAI caching" if self.enable_caching else "without caching"
            playback_status = "with audio playback" if self.enable_audio_playback else "silent"
            print(f"File processing ready: {self.duration:.2f}s ({loop_status}, {cache_status}, {playback_status})")
        else:
            print("No audio file loaded - file side will remain static")
        
        print("Left side: Real-time microphone input")
        print("Right side: Audio file processing")
        if self.loop_audio:
            print("Audio file will loop continuously")
        print("Press Ctrl+C to stop.")
        
        # Calculate animation interval - make it responsive to both sides
        real_time_interval = (self.chunk_size / self.sample_rate) * 1000  # ms
        # Adjust interval based on playback speed but keep it reasonable for real-time side
        animation_interval = max(10, int(real_time_interval / max(1, self.playback_speed)))
        
        print(f"Animation interval: {animation_interval}ms (chunk duration: {real_time_interval:.1f}ms)")
        
        # Start visualization
        self.animation = animation.FuncAnimation(
            self.fig, self.update_visualization, interval=animation_interval, 
            blit=False, cache_frame_data=False
        )
        plt.show()

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.playback_stream:
                self.playback_stream.stop_stream()
                self.playback_stream.close()
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
        print("Dual SAI system stopped.")

    def reset_file_position(self):
        """Manually reset the file position to the beginning"""
        self.current_position = 0
        self.chunks_processed = 0
        self.loop_count = 0
        # Reset cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        print("File position reset to beginning")

    def get_cache_stats(self):
        """Get current cache statistics"""
        if not self.enable_caching:
            return "Caching disabled"
        
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return "No cache requests yet"
        
        hit_rate = (self.cache_hits / total_requests) * 100
        cache_size = len(self.sai_cache)
        memory_mb = cache_size * self.chunk_size * 4 / 1024 / 1024  # Rough estimate
        
        return f"Cache: {cache_size} entries, {hit_rate:.1f}% hit rate, ~{memory_mb:.1f}MB"

    def clear_cache(self):
        """Clear the SAI cache to free memory"""
        if self.enable_caching:
            old_size = len(self.sai_cache)
            self.sai_cache.clear()
            self.cache_state_snapshots.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            print(f"Cache cleared: {old_size} entries removed")

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description='Dual SAI Visualization: Real-time + File Input')
    parser.add_argument('--audio-file', help='Path to audio file for right side processing')
    parser.add_argument('--chunk-size', type=int, default=512, help='Audio chunk size (default: 512)')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate (default: 16000)')
    parser.add_argument('--sai-width', type=int, default=400, help='SAI width (default: 400)')
    parser.add_argument('--whisper-model', default='tiny', help='Whisper model (default: tiny)')
    parser.add_argument('--whisper-interval', type=float, default=2.0, help='Whisper processing interval in seconds (default: 2.0)')
    parser.add_argument('--enable-translation', action='store_true', help='Enable translation')
    parser.add_argument('--from-lang', default='en', help='Source language for translation (default: en)')
    parser.add_argument('--to-lang', default='zh', help='Target language for translation (default: zh)')
    parser.add_argument('--speed', type=float, default=3.0, help='File playback speed multiplier (default: 3.0)')
    parser.add_argument('--no-loop', action='store_true', help='Disable audio file looping (default: looping enabled)')
    parser.add_argument('--no-cache', action='store_true', help='Disable SAI result caching (default: caching enabled)')
    parser.add_argument('--enable-playback', action='store_true', help='Enable audio playback of the file (default: silent)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Check if audio file exists (if provided)
    if args.audio_file and not os.path.exists(args.audio_file):
        print(f"Warning: Audio file '{args.audio_file}' not found")
        print("Continuing with real-time only mode...")
        args.audio_file = None
    
    try:
        processor = DualSAIProcessor(
            audio_file_path=args.audio_file,
            chunk_size=args.chunk_size,
            sample_rate=args.sample_rate,
            sai_width=args.sai_width,
            whisper_model=args.whisper_model,
            whisper_interval=args.whisper_interval,
            enable_translation=args.enable_translation,
            from_lang=args.from_lang,
            to_lang=args.to_lang,
            debug=args.debug,
            playback_speed=args.speed,
            loop_audio=not args.no_loop,  # Invert the no_loop flag
            enable_caching=not args.no_cache,  # Invert the no_cache flag
            enable_audio_playback=args.enable_playback
        )
        
        processor.start()
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error in dual SAI processing: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Example usage when run directly
    import sys
    if len(sys.argv) == 1:
        print("Dual SAI Visualization System")
        print("Usage examples:")
        print("  python script.py --debug")
        print("  python script.py --audio-file audio.wav --debug")
        print("  python script.py --audio-file audio.wav --speed 0.5 --whisper-model base")
        print("  python script.py --audio-file audio.wav --no-loop  # Disable looping")
        print("  python script.py --audio-file audio.wav --no-cache # Disable caching")
        print("  python script.py --audio-file audio.wav            # Uses default 3.0x speed")
        print("  python script.py --audio-file audio.wav --enable-playback  # With sound")
        print("Run with --help for all options")
        print("\nStarting with default settings (real-time only)...")
        processor = DualSAIProcessor(debug=True, loop_audio=True, enable_caching=True, playback_speed=3.0)
        try:
            processor.start()
        except KeyboardInterrupt:
            print("\nShutting down...")
            processor.stop()
    else:
        main()