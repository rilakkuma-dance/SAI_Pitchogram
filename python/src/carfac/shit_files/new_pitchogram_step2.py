import sys
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button
import matplotlib.font_manager as fm
import threading
import queue
import time
import torch
from dataclasses import dataclass
import librosa
import argparse
import os
import sounddevice as sd

sys.path.append('./jax')
import jax
import jax.numpy as jnp
import carfac.jax.carfac as carfac

from carfac.np.carfac import CarParams
import sai

plt.rcParams['font.sans-serif'] = [
    # Chinese fonts
    'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB',
    # Vietnamese fonts (most use Latin extended)
    'Arial Unicode MS', 'Tahoma', 'Times New Roman', 'Calibri',
    # Thai fonts
    'Tahoma', 'Arial Unicode MS', 'Leelawadee UI', 'Cordia New',
    # Fallback fonts
    'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # Handle minus signs properly

@dataclass
class SAIParams:
    """SAI Parameters"""
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
    """Calculates similarity scores between recorded voice and reference file SAI"""
    def __init__(self, history_size=50, smoothing_factor=0.3):
        self.history_size = history_size
        self.smoothing_factor = smoothing_factor
        
        self.file_sai_history = []  # Store file SAI frames for comparison
        self.recorded_sai_frames = []  # Store recorded SAI frames
        
        self.current_similarity = 0.0
        self.smoothed_similarity = 0.0
        self.max_similarity = 0.0
        self.similarity_history = []
        
    def add_file_frame(self, sai_frame):
        """Add a file SAI frame to history"""
        if len(self.file_sai_history) >= self.history_size:
            self.file_sai_history.pop(0)
        self.file_sai_history.append(sai_frame.copy())
    
    def set_recorded_frames(self, recorded_frames):
        """Set the recorded SAI frames for comparison"""
        self.recorded_sai_frames = recorded_frames

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

# ---------------- Dual SAI Visualization ----------------
class DualSAIVisualization:
    def __init__(self, audio_file_path=None, chunk_size=1024, sample_rate=16000, sai_width=200,
            debug=True, playback_speed=3.0, loop_audio=True, similarity_method='cosine'):

        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sai_width = sai_width
        self.debug = debug
        self.playback_speed = playback_speed
        self.loop_audio = loop_audio
        
        # Initialize reference text attributes
        self.reference_text = None           # Clean text for comparison
        self.reference_pronunciation = None  # Pronunciation guide for display
        self.translated_text = None          # Translation for display
        
        # Initialize similarity calculator
        self.similarity_calculator = SimilarityCalculator(history_size=50, smoothing_factor=0.2)
        self.similarity_method = similarity_method
             
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

        # Real-time audio setup
        self.audio_queue = queue.Queue(maxsize=50)
        
        # File processing setup
        self.audio_file_path = audio_file_path
        self.audio_data = None
        self.original_sr = None
        self.current_position = 0
        self.chunks_processed = 0
        self.duration = 0
        self.total_samples = 0
        self.loop_count = 0
        
        # Audio playback for reference file
        self.audio_playback_enabled = True
        self.audio_output_stream = None
        self.playback_position = 0.0  # Separate position for audio playback
        self.playback_chunk_size = chunk_size
        
        if audio_file_path and os.path.exists(audio_file_path):
            self._load_audio_file()
        
        # Audio stream
        self.p = None
        self.stream = None
        self.running = False
        
        # Setup visualization
        self._setup_dual_visualization()

    def _load_audio_file(self):
        """Load the audio file for file processing"""
        print(f"Loading audio file: {self.audio_file_path}")
        self.audio_data, self.original_sr = librosa.load(self.audio_file_path, sr=None)
        
        if self.original_sr != self.sample_rate:
            self.audio_data = librosa.resample(self.audio_data, orig_sr=self.original_sr, target_sr=self.sample_rate)
        
        if np.max(np.abs(self.audio_data)) > 0:
            self.audio_data = self.audio_data / np.max(np.abs(self.audio_data)) * 0.9
        
        self.total_samples = len(self.audio_data)
        self.duration = self.total_samples / self.sample_rate
        
        # Set reference text and pronunciation
        self.set_reference_text('谢谢')         # For comparison
        self.set_pronunciation_guide('xièxiè')   # For display
        self.set_translated_text('thank you')   # For display
        
        # Initialize audio playback
        if self.audio_playback_enabled:
            self._setup_audio_playback()

    def set_reference_text(self, text):
        """Set the reference text for comparison (clean text only)"""
        self.reference_text = text.strip()
        print(f"Reference text for comparison: {self.reference_text}")
        
        # Update display if we have both text and pronunciation
        self._update_display()

    def set_pronunciation_guide(self, pronunciation: str):
        """Set the pronunciation guide for display purposes"""
        self.reference_pronunciation = pronunciation
        print(f"Pronunciation guide: {pronunciation}")
        
        # Update display with both text and pronunciation
        self._update_display()

    def set_translated_text(self, translation: str):
        """Set the translation text for display purposes"""
        self.translated_text = translation.strip()
        print(f"Translation: {translation}")
        
        # Update display with reference, pronunciation, and translation
        self._update_display()

    def _update_display(self):
        """Update the display with reference text, pronunciation, and translation"""
        # Main reference line with pronunciation
        if self.reference_text and self.reference_pronunciation:
            main_display = f"{self.reference_text}({self.reference_pronunciation})"
        elif self.reference_text:
            main_display = f"{self.reference_text}"
        else:
            main_display = "No reference set"
        
        # Add translation on a new line if available
        if self.translated_text:
            full_display = f"{main_display}\n{self.translated_text}"
        else:
            full_display = main_display
        
        print(f"Display text: {full_display}")

    def _setup_audio_playback(self):
        """Setup audio playback for reference file"""
        try:
            # Using sounddevice for audio output
            self.audio_output_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.playback_chunk_size,
                callback=self._audio_playback_callback
            )
            print("Audio playback stream created")
        except Exception as e:
            print(f"Failed to create audio playback stream: {e}")
            self.audio_playback_enabled = False

    def _audio_playback_callback(self, outdata, frames, time, status):
        """Audio playback callback - synchronized with SAI processing"""
        if status:
            print(f"Audio playback status: {status}")
        
        try:
            if self.audio_data is not None and hasattr(self, 'playback_position'):
                # Use separate playback position to avoid conflicts with SAI processing
                start_pos = int(self.playback_position)
                end_pos = min(start_pos + frames, self.total_samples)
                
                if start_pos < self.total_samples:
                    chunk = self.audio_data[start_pos:end_pos]
                    
                    # Handle end of file
                    if len(chunk) < frames:
                        if self.loop_audio:
                            # Loop the audio
                            remaining = frames - len(chunk)
                            if remaining > 0 and self.total_samples > 0:
                                loop_chunk = self.audio_data[:min(remaining, self.total_samples)]
                                chunk = np.concatenate([chunk, loop_chunk])
                                self.playback_position = len(loop_chunk)
                            else:
                                self.playback_position = 0
                        else:
                            # Pad with silence
                            chunk = np.pad(chunk, (0, frames - len(chunk)), 'constant')
                            self.playback_position = end_pos
                    else:
                        self.playback_position = end_pos
                    
                    # Reset position if we've reached the end
                    if self.playback_position >= self.total_samples:
                        self.playback_position = 0
                    
                    outdata[:, 0] = chunk[:frames]
                else:
                    # Silence if no audio data
                    outdata.fill(0)
            else:
                outdata.fill(0)
                
        except Exception as e:
            print(f"Audio playback callback error: {e}")
            outdata.fill(0)

    def get_next_file_chunk(self):
        """Get next chunk from file with looping support"""
        if self.audio_data is None:
            return None, -1
        
        if self.current_position >= self.total_samples:
            if self.loop_audio:
                self.current_position = 0
                self.loop_count += 1
                if self.debug:
                    print(f"Audio file looped (loop #{self.loop_count})")
            else:
                return None, -1
        
        chunk_index = int(self.current_position / self.chunk_size)
        end_position = min(self.current_position + self.chunk_size, self.total_samples)
        
        chunk = self.audio_data[self.current_position:end_position]
        
        if len(chunk) < self.chunk_size:
            if self.loop_audio and self.current_position + len(chunk) >= self.total_samples:
                remaining_samples = self.chunk_size - len(chunk)
                if remaining_samples > 0 and self.total_samples > 0:
                    loop_start = self.audio_data[:min(remaining_samples, self.total_samples)]
                    chunk = np.concatenate([chunk, loop_start])
                    self.current_position = len(loop_start)
                else:
                    self.current_position = end_position
            else:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), 'constant')
                self.current_position = end_position
        else:
            self.current_position = end_position
        
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

        except Exception as e:
            print(f"Audio callback error: {e}")
        
        return (in_data, pyaudio.paContinue)

    def process_realtime_audio(self):
        """Process real-time audio stream"""
        print("Real-time audio processing thread started")
        while self.running:
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

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Real-time audio processing error: {e}")
                continue

    def _setup_dual_visualization(self):
        """Setup dual screen visualization with playback controls"""
        self.fig = plt.figure(figsize=(20, 16))
        gs = self.fig.add_gridspec(16, 2, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.8, 0.3, 0.3, 0.3, 0.3, 0.2])
        
        self.ax_realtime = self.fig.add_subplot(gs[0:10, 0])
        self.ax_file = self.fig.add_subplot(gs[0:10, 1])
        
        # Score display area
        self.ax_score = self.fig.add_subplot(gs[11, :])
        self.ax_score.set_facecolor('black')
        self.ax_score.set_xlim(0, 1)
        self.ax_score.set_ylim(0, 1)
        self.ax_score.axis('off')
        
        # Score display text
        self.score_display = self.ax_score.text(
            0.5, 0.5, 'Dual SAI Visualization - Microphone vs Reference Audio', 
            transform=self.ax_score.transAxes,
            horizontalalignment='center', verticalalignment='center',
            fontsize=14, color='white', weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8)
        )
        
        # Control buttons area
        self.ax_controls = self.fig.add_subplot(gs[12:16, :])
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
        self.ax_realtime.set_title("Live Microphone SAI", color='white', fontsize=14, pad=20)
        self.ax_realtime.axis('off')
        
        # Setup right side (file)
        self.im_file = self.ax_file.imshow(
            self.vis_file.img, aspect='auto', origin='upper',
            interpolation='bilinear', extent=[0, 200, 0, self.vis_file.output.shape[0]]
        )
        file_title = f"Reference Audio SAI: {os.path.basename(self.audio_file_path) if self.audio_file_path else 'No file loaded'}"
        self.ax_file.set_title(file_title, color='white', fontsize=14, pad=20)
        self.ax_file.axis('off')
        
        # Add text overlays for each side
        self.transcription_realtime = self.ax_realtime.text(
            0.02, 0.02, 'Listening... (speak into microphone)', transform=self.ax_realtime.transAxes,
            verticalalignment='bottom', fontsize=20, color='lime', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        
        self.transcription_file = self.ax_file.text(
            0.02, 0.02, '', transform=self.ax_file.transAxes,
            verticalalignment='bottom', fontsize=20, color='cyan', weight='bold',
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
        """Setup control buttons"""
        button_height = 0.05
        button_width = 0.1
        button_y = 0.02
        spacing = 0.15
        start_x = (1.0 - button_width) / 2  # Center the single button
        
        # Playback toggle button
        self.ax_playback = plt.axes([start_x, button_y, button_width, button_height], 
                                   facecolor='none', frameon=False)
        self.btn_playback = Button(self.ax_playback, 'Play Audio', 
                                  color=(0, 0.6, 0, 0.8), 
                                  hovercolor=(0, 0.8, 0, 1.0))
        self.btn_playback.label.set_fontsize(12)
        self.btn_playback.label.set_color('white')
        self.btn_playback.label.set_weight('bold')
        self.btn_playback.on_clicked(self.toggle_playback)

    def toggle_playback(self, event=None):
        """Toggle audio playback on/off"""
        if self.audio_playback_enabled and self.audio_output_stream:
            if self.audio_output_stream.active:
                self.audio_output_stream.stop()
                self.btn_playback.label.set_text('Play Audio')
                self.btn_playback.color = (0, 0.6, 0, 0.8)
                print("Audio playback stopped")
            else:
                self.audio_output_stream.start()
                self.btn_playback.label.set_text('Stop Audio')
                self.btn_playback.color = (0.8, 0, 0, 0.8)
                print("Audio playback started")
        else:
            print("Audio playback not available")

    def update_visualization(self, frame):
        """Update both visualizations"""
        try:
            # Update real-time side
            current_max_rt = np.max(self.vis_realtime.img) if self.vis_realtime.img.size else 1
            self.im_realtime.set_data(self.vis_realtime.img)
            self.im_realtime.set_clim(vmin=0, vmax=max(1, min(255, current_max_rt * 1.3)))
            
            # Update file side - Process multiple chunks if needed to maintain speed
            if self.audio_data is not None:
                target_chunks_per_frame = max(1, int(self.playback_speed))
                
                for _ in range(target_chunks_per_frame):
                    chunk, chunk_index = self.get_next_file_chunk()
                    if chunk is not None and chunk_index >= 0:
                        try:
                            # Process file chunk
                            nap_output = self.carfac_file.process_chunk(chunk)
                            sai_output = self.SAI_file.RunSegment(nap_output)
                            self.vis_file.get_vowel_embedding(nap_output)
                            vis_output = self.vis_file.run_frame(sai_output)

                            # Store file SAI frame for comparison
                            self.similarity_calculator.add_file_frame(sai_output)

                            self.vis_file.img[:, :-1] = self.vis_file.img[:, 1:]
                            self.vis_file.draw_column(self.vis_file.img[:, -1])

                            current_file_position = self.current_position
                            if self.current_position == 0 and self.chunks_processed > 0:
                                current_time = self.duration
                            else:
                                current_time = (current_file_position / self.total_samples) * self.duration if self.total_samples > 0 else 0
                            
                            self.chunks_processed += 1
                            
                        except Exception as e:
                            print(f"Error processing file chunk {chunk_index}: {e}")
                            break
                    else:
                        break

            current_max_file = np.max(self.vis_file.img) if self.vis_file.img.size else 1
            self.im_file.set_data(self.vis_file.img)
            self.im_file.set_clim(vmin=0, vmax=max(1, min(255, current_max_file * 1.3)))
            
            # Update text displays - simplified without Whisper
            # Display static text for reference
            reference_display = ''
            if self.reference_text and self.reference_pronunciation:
                reference_display = f"{self.reference_text}({self.reference_pronunciation})"
            elif self.reference_text:
                reference_display = f"{self.reference_text}"
            else:
                reference_display = "No reference set"
            
            if self.translated_text:
                reference_display += f"\n{self.translated_text}"
            
            self.transcription_file.set_text(reference_display)
            
            # Update progress
            if self.audio_data is not None:
                progress_pct = (self.current_position / self.total_samples) * 100 if self.total_samples > 0 else 0
                current_time = (self.current_position / self.total_samples) * self.duration if self.total_samples > 0 else 0
                loop_info = f" (Loop #{self.loop_count})" if self.loop_count > 0 and self.loop_audio else ""
                progress_text = f"Progress: {progress_pct:.1f}% | {current_time:.1f}s/{self.duration:.1f}s{loop_info}"
                self.progress_file.set_text(progress_text)
            
            # Update status
            if hasattr(self, 'audio_queue') and not self.audio_queue.empty():
                queue_size = self.audio_queue.qsize()
                self.status_realtime.set_text(f"Audio active (queue: {queue_size})")
                self.status_realtime.set_color('white')
            else:
                self.status_realtime.set_text("Listening for audio")
                self.status_realtime.set_color('white')
            
        except Exception as e:
            print(f"Visualization update error: {e}")
        
        return [self.im_realtime, self.im_file, self.transcription_realtime, 
                self.transcription_file, self.progress_file, self.status_realtime,
                self.score_display]

    def start(self):
        self.p = pyaudio.PyAudio()
        
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
        except Exception as e:
            print(f"Failed to open audio stream: {e}")
            return

        self.running = True
        threading.Thread(target=self.process_realtime_audio, daemon=True).start()
        
        # Start input stream
        if self.stream:
            self.stream.start_stream()
            print("Audio input stream started")
        
        # Start output stream for audio playback
        if self.audio_playback_enabled and self.audio_output_stream:
            self.audio_output_stream.start()
            print("Audio playback stream started")
        
        real_time_interval = (self.chunk_size / self.sample_rate) * 1000
        animation_interval = max(10, int(real_time_interval / max(1, self.playback_speed)))
                
        self.animation = animation.FuncAnimation(
            self.fig, self.update_visualization, interval=animation_interval, 
            blit=False, cache_frame_data=False
        )
        plt.show()

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        # Stop audio playback
        if self.audio_output_stream:
            try:
                self.audio_output_stream.stop()
                self.audio_output_stream.close()
            except:
                pass
        
        # Stop audio input
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

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description='Dual SAI Visualization')
    parser.add_argument('--audio-file', default='reference/mandarin_shu.mp3', 
                    help='Path to reference audio file')
    parser.add_argument('--chunk-size', type=int, default=512, help='Audio chunk size (default: 512)')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate (default: 16000)')
    parser.add_argument('--sai-width', type=int, default=400, help='SAI width (default: 400)')
    parser.add_argument('--similarity-method', default='cosine', choices=['cosine', 'correlation', 'euclidean', 'spectral'], 
                        help='Similarity calculation method (default: cosine)')
    parser.add_argument('--speed', type=float, default=3.0, help='File playback speed multiplier (default: 3.0)')
    parser.add_argument('--no-loop', action='store_true', help='Disable audio file looping (default: looping enabled)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.audio_file and not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found")
        return 1
    
    try:
        processor = DualSAIVisualization(
            audio_file_path=args.audio_file,
            chunk_size=args.chunk_size,
            sample_rate=args.sample_rate,
            sai_width=args.sai_width,
            debug=args.debug,
            playback_speed=args.speed,
            loop_audio=not args.no_loop,
            similarity_method=args.similarity_method,
        )
        
        processor.start()
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 0
    except Exception as e:
        print(f"Error in dual SAI visualization: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    import os
    if len(sys.argv) == 1:
        # Get the script's directory (src)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Audio file is in reference subdirectory
        default_audio = os.path.join(script_dir, 'reference', 'mandarin_thankyou.mp3')
        
        if os.path.exists(default_audio):
            sys.argv.append('--audio-file')
            sys.argv.append(default_audio)
        else:
            sys.exit(1)
    sys.exit(main() or 0)