#!/usr/bin/env python3
"""
Dual SAI Pronunciation Trainer with Personalized Formant Calibration
Complete single-file implementation with voice-specific color mapping
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import threading
import time
import json
import os
import argparse
import librosa
import scipy.signal
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter
import pystoi
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import your existing modules (assuming they exist)
# from carfac_processor import RealCARFACProcessor
# from sai_processor import sai
# from whisper_processor import WhisperRealTimeProcessor
# from text_similarity import TextSimilarityCalculator
# from audio_playback import AudioPlayback

class PersonalizedVowelMatrix:
    """Personalized vowel formant calibration system"""
    
    def __init__(self):
        self.calibration_words = ["bee", "bay", "bad", "law", "bow", "boo"]
        self.target_vowels = {
            "bee": ("i", [250, 2300]),    # High front
            "bay": ("eɪ", [400, 2100]),   # Mid front  
            "bad": ("æ", [650, 1700]),    # Low front
            "law": ("ɔ", [550, 850]),     # Mid back
            "bow": ("oʊ", [400, 950]),    # Mid back rounded
            "boo": ("u", [300, 870])      # High back
        }
        
        self.vowel_samples = {}
        self.formant_data = {}
        self.is_calibrated = False
        self.formant_boundaries = None
        
    def add_vowel_sample(self, word: str, sai_frames: np.ndarray, sample_rate: int) -> bool:
        """Add a vowel sample and extract formant frequencies"""
        if word not in self.calibration_words:
            return False
            
        try:
            # Convert SAI frames back to audio for formant analysis
            # This is a simplified conversion - you may need to adjust based on your SAI implementation
            audio_data = self._sai_to_audio_estimate(sai_frames, sample_rate)
            
            # Extract formant frequencies
            f1, f2 = self._extract_formants(audio_data, sample_rate)
            
            if f1 is not None and f2 is not None:
                self.vowel_samples[word] = {
                    'f1': f1,
                    'f2': f2,
                    'sai_frames': sai_frames
                }
                print(f"Captured {word}: F1={f1:.0f}Hz, F2={f2:.0f}Hz")
                return True
                
        except Exception as e:
            print(f"Error processing vowel sample for '{word}': {e}")
            
        return False
    
    def _sai_to_audio_estimate(self, sai_frames: np.ndarray, sample_rate: int) -> np.ndarray:
        """Estimate audio signal from SAI frames for formant analysis"""
        # This is a simplified approach - you may need to adapt based on your SAI structure
        if len(sai_frames.shape) == 3:  # [time, channels, delay]
            # Sum across delay dimension and take first channel
            audio_estimate = np.sum(sai_frames[:, 0, :], axis=1)
        else:
            # Handle different SAI frame structures
            audio_estimate = np.mean(sai_frames, axis=-1) if len(sai_frames.shape) > 1 else sai_frames
            
        # Normalize
        if np.max(np.abs(audio_estimate)) > 0:
            audio_estimate = audio_estimate / np.max(np.abs(audio_estimate))
            
        return audio_estimate.astype(np.float32)
    
    def _extract_formants(self, audio: np.ndarray, sample_rate: int) -> Tuple[Optional[float], Optional[float]]:
        """Extract F1 and F2 formant frequencies using LPC analysis"""
        try:
            # Pre-emphasis
            pre_emphasis = 0.97
            emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
            
            # Window the signal (middle portion for stable vowel)
            window_size = min(len(emphasized), int(0.025 * sample_rate))  # 25ms window
            start = max(0, len(emphasized) // 2 - window_size // 2)
            windowed = emphasized[start:start + window_size]
            
            # Apply Hamming window
            windowed = windowed * np.hamming(len(windowed))
            
            # LPC analysis (order based on sample rate)
            lpc_order = int(sample_rate / 1000) + 2  # Rule of thumb: 1 coefficient per kHz + 2
            lpc_order = min(lpc_order, len(windowed) - 1)
            
            if lpc_order < 2:
                return None, None
                
            # Calculate LPC coefficients
            lpc_coeffs = self._lpc_analysis(windowed, lpc_order)
            
            # Find formant frequencies from LPC roots
            formants = self._lpc_to_formants(lpc_coeffs, sample_rate)
            
            if len(formants) >= 2:
                # Return first two formants (F1, F2)
                return formants[0], formants[1]
                
        except Exception as e:
            print(f"Formant extraction error: {e}")
            
        return None, None
    
    def _lpc_analysis(self, signal: np.ndarray, order: int) -> np.ndarray:
        """Linear Predictive Coding analysis"""
        # Autocorrelation method
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        
        # Solve Yule-Walker equations using Levinson-Durbin algorithm
        r = autocorr[:order + 1]
        
        # Initialize
        a = np.zeros(order + 1)
        a[0] = 1.0
        error = r[0]
        
        for i in range(1, order + 1):
            # Calculate reflection coefficient
            lambda_i = 0
            for j in range(i):
                lambda_i += a[j] * r[i - j]
            lambda_i = -lambda_i / error
            
            # Update coefficients
            a_new = a.copy()
            for j in range(1, i):
                a_new[j] = a[j] + lambda_i * a[i - j]
            a_new[i] = lambda_i
            
            a = a_new
            error = error * (1 - lambda_i**2)
            
        return a
    
    def _lpc_to_formants(self, lpc_coeffs: np.ndarray, sample_rate: int) -> List[float]:
        """Convert LPC coefficients to formant frequencies"""
        # Find roots of LPC polynomial
        roots = np.roots(lpc_coeffs)
        
        # Keep only roots inside unit circle with positive imaginary parts
        formants = []
        for root in roots:
            if abs(root) < 1.0 and np.imag(root) > 0:
                # Convert to frequency
                frequency = np.angle(root) * sample_rate / (2 * np.pi)
                if 50 < frequency < sample_rate / 2:  # Valid frequency range
                    formants.append(frequency)
        
        # Sort by frequency
        formants.sort()
        
        # Filter to typical formant ranges
        valid_formants = []
        for f in formants:
            if 200 < f < 4000:  # Typical formant range
                valid_formants.append(f)
                
        return valid_formants[:5]  # Return first 5 formants
    
    def calibrate_formant_boundaries(self) -> bool:
        """Calculate personalized formant boundaries from collected samples"""
        if len(self.vowel_samples) < 4:  # Need at least 4 vowels for calibration
            print(f"Not enough vowel samples for calibration: {len(self.vowel_samples)}/6")
            return False
        
        try:
            # Extract all F1 and F2 values
            f1_values = [data['f1'] for data in self.vowel_samples.values()]
            f2_values = [data['f2'] for data in self.vowel_samples.values()]
            
            # Calculate boundaries with some padding
            f1_min, f1_max = min(f1_values), max(f1_values)
            f2_min, f2_max = min(f2_values), max(f2_values)
            
            # Add padding (10% on each side)
            f1_padding = (f1_max - f1_min) * 0.1
            f2_padding = (f2_max - f2_min) * 0.1
            
            self.formant_boundaries = {
                'f1_lo': max(200, f1_min - f1_padding),
                'f1_hi': min(1000, f1_max + f1_padding),
                'f2_lo': max(800, f2_min - f2_padding),
                'f2_hi': min(3000, f2_max + f2_padding)
            }
            
            self.is_calibrated = True
            
            print("Formant boundaries calculated:")
            print(f"  F1: {self.formant_boundaries['f1_lo']:.0f} - {self.formant_boundaries['f1_hi']:.0f} Hz")
            print(f"  F2: {self.formant_boundaries['f2_lo']:.0f} - {self.formant_boundaries['f2_hi']:.0f} Hz")
            
            return True
            
        except Exception as e:
            print(f"Error calculating formant boundaries: {e}")
            return False
    
    def get_formant_boundaries(self) -> Optional[Dict[str, float]]:
        """Get the calculated formant boundaries"""
        return self.formant_boundaries if self.is_calibrated else None
    
    def save_calibration(self, filepath: str) -> bool:
        """Save calibration data to file"""
        if not self.is_calibrated:
            return False
            
        try:
            data = {
                'formant_boundaries': self.formant_boundaries,
                'vowel_samples': {
                    word: {'f1': data['f1'], 'f2': data['f2']} 
                    for word, data in self.vowel_samples.items()
                },
                'is_calibrated': self.is_calibrated
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"Formant calibration saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False
    
    def load_calibration(self, filepath: str) -> bool:
        """Load calibration data from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.formant_boundaries = data.get('formant_boundaries')
            self.is_calibrated = data.get('is_calibrated', False)
            
            # Reconstruct vowel samples (without SAI frames)
            vowel_data = data.get('vowel_samples', {})
            for word, formants in vowel_data.items():
                self.vowel_samples[word] = {
                    'f1': formants['f1'],
                    'f2': formants['f2'],
                    'sai_frames': None  # Not saved
                }
            
            print(f"Formant calibration loaded from {filepath}")
            return self.is_calibrated
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False


class PersonalizedVisualizationHandler:
    """Visualization handler with personalized formant-based coloring"""
    
    def __init__(self, sample_rate: int, n_channels: int, formant_calibrator: PersonalizedVowelMatrix):
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.formant_calibrator = formant_calibrator
        
        # Create personalized color mapping
        self.create_personalized_colormap()
    
    def create_personalized_colormap(self):
        """Create color mapping based on user's formant calibration"""
        if self.formant_calibrator.is_calibrated:
            boundaries = self.formant_calibrator.get_formant_boundaries()
            
            # Use personalized boundaries for color mapping
            self.f1_range = (boundaries['f1_lo'], boundaries['f1_hi'])
            self.f2_range = (boundaries['f2_lo'], boundaries['f2_hi'])
            
            print(f"Using personalized color mapping:")
            print(f"  F1 range: {self.f1_range[0]:.0f}-{self.f1_range[1]:.0f} Hz")
            print(f"  F2 range: {self.f2_range[0]:.0f}-{self.f2_range[1]:.0f} Hz")
        else:
            # Use default ranges
            self.f1_range = (250, 850)
            self.f2_range = (850, 2500)
            print("Using default color mapping ranges")
    
    def frequency_to_color(self, f1: float, f2: float) -> Tuple[float, float, float]:
        """Convert formant frequencies to RGB color using personalized mapping"""
        # Normalize frequencies to [0, 1] based on personalized ranges
        f1_norm = np.clip((f1 - self.f1_range[0]) / (self.f1_range[1] - self.f1_range[0]), 0, 1)
        f2_norm = np.clip((f2 - self.f2_range[0]) / (self.f2_range[1] - self.f2_range[0]), 0, 1)
        
        # Create color mapping
        # F1 (vertical position) -> Red/Green balance
        # F2 (horizontal position) -> Blue component
        red = 1.0 - f1_norm  # High F1 = less red (darker)
        green = f1_norm      # High F1 = more green
        blue = f2_norm       # High F2 = more blue
        
        return (red, green, blue)
    
    def sai_to_formant_colors(self, sai_frame: np.ndarray) -> np.ndarray:
        """Convert SAI frame to color representation using personalized mapping"""
        # This is a simplified approach - you'll need to adapt based on your SAI structure
        # Extract dominant frequencies from SAI frame
        
        if len(sai_frame.shape) == 2:
            height, width = sai_frame.shape
        else:
            height, width = sai_frame.shape[:2]
        
        # Create color array
        color_frame = np.zeros((height, width, 3))
        
        # Convert SAI channels to approximate formant frequencies
        for i in range(height):
            for j in range(width):
                # Map SAI coordinates to formant space
                # This mapping depends on your SAI implementation
                f1_approx = self.f1_range[0] + (i / height) * (self.f1_range[1] - self.f1_range[0])
                f2_approx = self.f2_range[0] + (j / width) * (self.f2_range[1] - self.f2_range[0])
                
                # Get color for this formant combination
                color = self.frequency_to_color(f1_approx, f2_approx)
                
                # Weight by SAI magnitude
                magnitude = np.abs(sai_frame[i, j]) if len(sai_frame.shape) == 2 else np.abs(sai_frame[i, j, 0])
                color_frame[i, j] = [c * magnitude for c in color]
        
        return color_frame


class DualSAIWithRecording:
    """Enhanced SAI processor with personalized formant calibration"""
    
    def __init__(self, audio_file_path=None, chunk_size=1024, sample_rate=16000, sai_width=200,
                 whisper_model="base", whisper_interval=1.5, debug=True, playback_speed=3.0, 
                 loop_audio=True, similarity_method='cosine', save_recordings=True, recording_dir="recordings",
                 language: str = "en"):
        
        # Core parameters
        self.audio_file_path = audio_file_path
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sai_width = sai_width
        self.whisper_model = whisper_model
        self.whisper_interval = whisper_interval
        self.debug = debug
        self.playback_speed = playback_speed
        self.loop_audio = loop_audio
        self.similarity_method = similarity_method
        self.save_recordings = save_recordings
        self.recording_dir = recording_dir
        self.language = language
        
        # Recording state
        self.is_recording = False
        self.recorded_audio = []
        self.recording_start_time = None
        self.recording_duration = 3.0  # seconds
        
        # Audio processing
        self.n_channels = 64  # Default CARFAC channels
        
        # SAI parameters
        self.sai_params = {
            'width': sai_width,
            'height': self.n_channels
        }
        
        # Add formant calibration system
        self.formant_calibrator = PersonalizedVowelMatrix()
        self.formant_calibration_mode = False
        self.current_formant_word_index = 0
        self.needs_formant_calibration = True
        self.formant_words = self.formant_calibrator.calibration_words
        
        # Initialize processors (you'll need to uncomment these when you have the modules)
        # self.carfac_realtime = RealCARFACProcessor(fs=sample_rate, n_ch=self.n_channels)
        # self.carfac_file = RealCARFACProcessor(fs=sample_rate, n_ch=self.n_channels)
        # self.sai_realtime = sai.SAI(self.sai_params)
        # self.sai_file = sai.SAI(self.sai_params)
        # self.whisper_realtime = WhisperRealTimeProcessor(model_size=whisper_model, device="cpu")
        
        # Text processing
        self.file_transcription = ""
        self.text_similarity_calculator = None
        
        # Visualization setup
        self.setup_visualization()
        
        # Initially create default visualization handlers
        self.vis_realtime = VisualizationHandler(sample_rate, sai_params=self.sai_params) if 'VisualizationHandler' in globals() else None
        self.vis_file = VisualizationHandler(sample_rate, sai_params=self.sai_params) if 'VisualizationHandler' in globals() else None
        
        # Audio playback
        self.audio_playback_enabled = True
        # self.audio_playback = AudioPlayback(sample_rate) if audio_playback_enabled else None
        
        # Load audio file if provided
        if audio_file_path:
            self._load_audio_file()
    
    def setup_visualization(self):
        """Setup matplotlib visualization"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Dual SAI Pronunciation Trainer with Personalized Voice Calibration', 
                         fontsize=16, color='white')
        
        # Create subplots
        gs = self.fig.add_gridspec(3, 3, height_ratios=[1, 2, 2], width_ratios=[1, 1, 1])
        
        # Control panel
        self.ax_controls = self.fig.add_subplot(gs[0, :])
        self.ax_controls.set_title('Controls & Status', color='white')
        self.ax_controls.axis('off')
        
        # SAI visualizations
        self.ax_sai_file = self.fig.add_subplot(gs[1, 0])
        self.ax_sai_file.set_title('Reference Audio SAI', color='white')
        
        self.ax_sai_realtime = self.fig.add_subplot(gs[1, 1])
        self.ax_sai_realtime.set_title('Your Pronunciation SAI', color='white')
        
        self.ax_comparison = self.fig.add_subplot(gs[1, 2])
        self.ax_comparison.set_title('Formant Comparison', color='white')
        
        # Waveform and spectrogram
        self.ax_waveform = self.fig.add_subplot(gs[2, :2])
        self.ax_waveform.set_title('Audio Waveform', color='white')
        
        self.ax_spectrogram = self.fig.add_subplot(gs[2, 2])
        self.ax_spectrogram.set_title('Spectrogram', color='white')
        
        # Add control elements
        self.add_control_elements()
        
        plt.tight_layout()
    
    def add_control_elements(self):
        """Add buttons and text displays"""
        # Record button
        ax_record = plt.axes([0.1, 0.02, 0.1, 0.04])
        self.btn_record = Button(ax_record, 'Record', color=(0, 0.8, 1, 0.8), hovercolor=(0, 0.6, 1, 1))
        self.btn_record.on_clicked(self.start_recording)
        
        # Play button
        ax_play = plt.axes([0.25, 0.02, 0.1, 0.04])
        self.btn_play = Button(ax_play, 'Play Reference', color=(0, 1, 0, 0.8), hovercolor=(0, 0.8, 0, 1))
        self.btn_play.on_clicked(self.play_reference_audio)
        
        # Calibration button
        ax_calibrate = plt.axes([0.4, 0.02, 0.15, 0.04])
        self.btn_calibrate = Button(ax_calibrate, 'Voice Calibration', color=(1, 0.6, 0, 0.8), hovercolor=(1, 0.4, 0, 1))
        self.btn_calibrate.on_clicked(lambda x: self.start_formant_calibration())
        
        # Skip calibration button
        ax_skip = plt.axes([0.58, 0.02, 0.12, 0.04])
        self.btn_skip = Button(ax_skip, 'Skip Calibration', color=(0.5, 0.5, 0.5, 0.8), hovercolor=(0.7, 0.7, 0.7, 1))
        self.btn_skip.on_clicked(lambda x: self.skip_formant_calibration())
        
        # Status displays
        self.score_display = self.ax_controls.text(0.02, 0.7, 'Ready - Click "Voice Calibration" to start', 
                                                  transform=self.ax_controls.transAxes, fontsize=14, 
                                                  color='white', weight='bold')
        
        self.status_realtime = self.ax_controls.text(0.02, 0.4, 'Status: Ready', 
                                                    transform=self.ax_controls.transAxes, fontsize=12, 
                                                    color='white')
        
        self.transcription_realtime = self.ax_controls.text(0.02, 0.1, 'Transcription: (none)', 
                                                           transform=self.ax_controls.transAxes, fontsize=12, 
                                                           color='gray')
    
    def start_formant_calibration(self):
        """Start the real-time formant calibration process"""
        if self.formant_calibrator.is_calibrated:
            print("Formant calibration already completed")
            return
        
        self.formant_calibration_mode = True
        self.current_formant_word_index = 0
        
        print("\n=== Voice Formant Calibration ===")
        print("Say each word clearly to calibrate colors to your voice.")
        print("This analyzes your vocal tract characteristics.")
        print(f"Words: {' | '.join(self.formant_words)}")
        
        self._prompt_formant_word()
    
    def _prompt_formant_word(self):
        """Show the current word for formant calibration"""
        if self.current_formant_word_index < len(self.formant_words):
            word = self.formant_words[self.current_formant_word_index]
            progress = f"({self.current_formant_word_index + 1}/{len(self.formant_words)})"
            
            print(f"\nSay: '{word}' {progress}")
            
            # Update the score display
            if hasattr(self, 'score_display'):
                self.score_display.set_text(f"Formant Calibration: Say '{word}' {progress}")
                self.score_display.set_color('orange')
            
            # Update recording button
            if hasattr(self, 'btn_record'):
                self.btn_record.label.set_text(f"Say '{word}'")
                self.btn_record.color = (1, 0.6, 0, 0.8)  # Orange for formant calibration
        else:
            self._finish_formant_calibration()
    
    def _finish_formant_calibration(self):
        """Complete formant calibration and update visualization"""
        print("\nAnalyzing your voice characteristics...")
        
        if self.formant_calibrator.calibrate_formant_boundaries():
            # Create new personalized visualization handlers
            self.vis_realtime = PersonalizedVisualizationHandler(
                self.sample_rate, self.n_channels, self.formant_calibrator
            )
            self.vis_file = PersonalizedVisualizationHandler(
                self.sample_rate, self.n_channels, self.formant_calibrator
            )
            
            # Save calibration for future use
            self.formant_calibrator.save_calibration("formant_calibration.json")
            
            print("Formant calibration complete! Colors now match your voice.")
            
            # Show calibration results
            boundaries = self.formant_calibrator.get_formant_boundaries()
            print(f"Your voice characteristics:")
            print(f"  F1 range: {boundaries['f1_lo']:.0f}-{boundaries['f1_hi']:.0f} Hz")
            print(f"  F2 range: {boundaries['f2_lo']:.0f}-{boundaries['f2_hi']:.0f} Hz")
            
            # Update display
            if hasattr(self, 'score_display'):
                self.score_display.set_text("Voice calibrated! Colors optimized for your vocal tract.")
                self.score_display.set_color('lime')
            
            # Reset recording button
            if hasattr(self, 'btn_record'):
                self.btn_record.label.set_text('Record')
                self.btn_record.color = (0, 0.8, 1, 0.8)
        else:
            print("Formant calibration failed. Using default voice mapping.")
            if hasattr(self, 'score_display'):
                self.score_display.set_text("Calibration failed - using default colors")
                self.score_display.set_color('yellow')
        
        self.formant_calibration_mode = False
        self.current_formant_word_index = 0
    
    def start_recording(self, event=None):
        """Modified to handle formant calibration"""
        if self.is_recording:
            self.stop_recording()
            return
        
        # Check if we need formant calibration first
        if self.needs_formant_calibration and not self.formant_calibrator.is_calibrated:
            if not self.formant_calibration_mode:
                self.start_formant_calibration()
                return
        
        # Normal recording (either calibration or practice)
        self.is_recording = True
        self.recorded_audio = []
        self.recording_start_time = time.time()
        
        # Update button appearance
        if self.formant_calibration_mode:
            self.btn_record.label.set_text('Recording...')
            self.btn_record.color = (1, 0, 0, 0.8)  # Red when recording
        else:
            self.btn_record.label.set_text('Recording...')
            self.btn_record.color = (1, 0, 0, 0.8)
        
        # Start countdown thread
        threading.Thread(target=self._recording_countdown, daemon=True).start()
        
        # Start audio capture (placeholder)
        print(f"Starting recording for {self.recording_duration} seconds...")
    
    def _recording_countdown(self):
        """Countdown timer for recording"""
        time.sleep(self.recording_duration)
        if self.is_recording:
            self.stop_recording()
    
    def stop_recording(self):
        """Stop recording and process audio"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        recording_duration = time.time() - self.recording_start_time
        
        print(f"Recording stopped after {recording_duration:.1f} seconds")
        
        # Reset button
        if self.formant_calibration_mode:
            word = self.formant_words[self.current_formant_word_index]
            self.btn_record.label.set_text(f"Say '{word}'")
            self.btn_record.color = (1, 0.6, 0, 0.8)
        else:
            self.btn_record.label.set_text('Record')
            self.btn_record.color = (0, 0.8, 1, 0.8)
        
        # Process the recorded audio
        threading.Thread(target=self._process_recorded_audio, daemon=True).start()
    
    def _process_recorded_audio(self):
        """Modified to handle both formant calibration and normal recordings"""
        try:
            print("Processing recorded audio...")
            
            # Convert to numpy array and ensure correct format
            recorded_array = np.array(self.recorded_audio, dtype=np.float32)
            
            if len(recorded_array) < self.sample_rate * 0.5:
                print("Recording too short for analysis")
                self._handle_short_recording()
                return
            
            # Normalize audio
            if np.max(np.abs(recorded_array)) > 0:
                recorded_array = recorded_array / np.max(np.abs(recorded_array)) * 0.9
            
            # Process through CARFAC and SAI (placeholder - uncomment when modules available)
            # temp_carfac = RealCARFACProcessor(fs=self.sample_rate)
            # temp_sai = sai.SAI(self.sai_params)
            
            recorded_sai_frames = []
            chunk_size = self.chunk_size
            
            # Placeholder SAI processing
            for i in range(0, len(recorded_array), chunk_size):
                chunk = recorded_array[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
                
                # nap_output = temp_carfac.process_chunk(chunk)
                # sai_output = temp_sai.RunSegment(nap_output)
                # For now, create dummy SAI output
                sai_output = np.random.random((self.n_channels, self.sai_width)) * np.max(np.abs(chunk))
                recorded_sai_frames.append(sai_output)
            
            # Handle formant calibration vs normal recording
            if self.formant_calibration_mode:
                self._process_formant_calibration_recording(recorded_sai_frames)
            else:
                self._process_normal_recording(recorded_array, recorded_sai_frames)
                
        except Exception as e:
            print(f"Error processing recorded audio: {e}")
            self._handle_processing_error()
    
    def _process_formant_calibration_recording(self, sai_frames):
        """Process a formant calibration recording"""
        word = self.formant_words[self.current_formant_word_index]
        
        # Add the vowel sample to formant calibrator
        success = self.formant_calibrator.add_vowel_sample(
            word, np.array(sai_frames), self.sample_rate
        )
        
        if success:
            print(f"✓ Analyzed formants for '{word}'")
            self.current_formant_word_index += 1
            
            # Move to next word or finish
            if self.current_formant_word_index < len(self.formant_words):
                self._prompt_formant_word()
            else:
                self._finish_formant_calibration()
        else:
            print(f"Failed to analyze '{word}', please try again")
            if hasattr(self, 'score_display'):
                self.score_display.set_text(f"Failed to analyze '{word}' - try again")
                self.score_display.set_color('red')
    
    def _process_normal_recording(self, audio_data, sai_frames):
        """Process a normal pronunciation practice recording"""
        # Update status
        self.status_realtime.set_text("Processing...")
        self.status_realtime.set_color('yellow')
        
        # Placeholder transcription (uncomment when WhisperRealTimeProcessor available)
        # if self.text_similarity_calculator:
        #     transcription = self.whisper_realtime.transcribe_once(audio_data, language=self.language)
        
        # For now, simulate transcription
        transcription = "hello"  # Placeholder
        
        if transcription and len(transcription.strip()) > 0:
            print(f"Recorded transcription: '{transcription}'")
            print(f"Reference text: '{self.file_transcription}'")
            
            # Placeholder similarity calculation
            distance = 0 if transcription.lower().strip() == self.file_transcription.lower().strip() else 1
            print(f"Edit distance: {distance}")
            
            # Calculate accuracy percentage
            ref_length = len(self.file_transcription) if self.file_transcription else 1
            error_percentage = (distance / ref_length) * 100
            accuracy_percentage = max(0, 100 - error_percentage)
            
            # Display results with improved scoring
            if distance == 0:
                self.transcription_realtime.set_color('lime')
                self.score_display.set_text("PERFECT MATCH!")
                self.score_display.set_color('lime')
            elif accuracy_percentage >= 85:
                self.transcription_realtime.set_color('lime')
                self.score_display.set_text(f"EXCELLENT! {accuracy_percentage:.0f}% accuracy")
                self.score_display.set_color('lime')
            elif accuracy_percentage >= 70:
                self.transcription_realtime.set_color('orange')
                self.score_display.set_text(f"GOOD! {accuracy_percentage:.0f}% accuracy")
                self.score_display.set_color('orange')
            elif accuracy_percentage >= 50:
                self.transcription_realtime.set_color('yellow')
                self.score_display.set_text(f"FAIR - {accuracy_percentage:.0f}% accuracy")
                self.score_display.set_color('yellow')
            else:
                self.transcription_realtime.set_color('red')
                self.score_display.set_text(f"NEEDS PRACTICE - {accuracy_percentage:.0f}% accuracy")
                self.score_display.set_color('red')
                
            # Update transcription display
            self.transcription_realtime.set_text(f"You said: '{transcription}'")
        else:
            self.transcription_realtime.set_color('gray')
            self.transcription_realtime.set_text("Transcription: (no speech detected)")
            self.score_display.set_text("No speech detected - Please try again")
            self.score_display.set_color('gray')
        
        # Update status back to ready
        self.status_realtime.set_text("Ready to record")
        self.status_realtime.set_color('white')
        
        # Update visualization with new SAI data
        self._update_sai_visualization(sai_frames)
    
    def _update_sai_visualization(self, sai_frames):
        """Update the SAI visualization with new data"""
        if sai_frames and self.vis_realtime:
            try:
                # Average across frames for display
                avg_sai = np.mean(sai_frames, axis=0)
                
                # Convert to color using personalized mapping
                color_frame = self.vis_realtime.sai_to_formant_colors(avg_sai)
                
                # Display in real-time SAI plot
                self.ax_sai_realtime.clear()
                self.ax_sai_realtime.imshow(color_frame, aspect='auto', origin='lower')
                self.ax_sai_realtime.set_title('Your Pronunciation SAI', color='white')
                self.ax_sai_realtime.set_xlabel('Time (samples)')
                self.ax_sai_realtime.set_ylabel('Frequency Channel')
                
                # Update comparison plot
                self._update_comparison_plot()
                
                plt.draw()
                
            except Exception as e:
                print(f"Error updating SAI visualization: {e}")
    
    def _update_comparison_plot(self):
        """Update the formant comparison plot"""
        try:
            self.ax_comparison.clear()
            
            if self.formant_calibrator.is_calibrated:
                # Plot calibrated formant space
                boundaries = self.formant_calibrator.get_formant_boundaries()
                
                # Draw formant boundaries
                rect = patches.Rectangle(
                    (boundaries['f2_lo'], boundaries['f1_lo']),
                    boundaries['f2_hi'] - boundaries['f2_lo'],
                    boundaries['f1_hi'] - boundaries['f1_lo'],
                    linewidth=2, edgecolor='lime', facecolor='none', alpha=0.7
                )
                self.ax_comparison.add_patch(rect)
                
                # Plot calibration vowels
                for word, data in self.formant_calibrator.vowel_samples.items():
                    self.ax_comparison.scatter(data['f2'], data['f1'], 
                                             s=100, alpha=0.8, label=word)
                
                self.ax_comparison.set_xlabel('F2 (Hz)', color='white')
                self.ax_comparison.set_ylabel('F1 (Hz)', color='white')
                self.ax_comparison.set_title('Calibrated Formant Space', color='white')
                self.ax_comparison.legend()
                self.ax_comparison.grid(True, alpha=0.3)
                
                # Invert F1 axis (traditional formant plot)
                self.ax_comparison.invert_yaxis()
            else:
                self.ax_comparison.text(0.5, 0.5, 'Voice calibration\nneeded', 
                                      ha='center', va='center', transform=self.ax_comparison.transAxes,
                                      fontsize=14, color='orange')
                self.ax_comparison.set_title('Formant Comparison', color='white')
            
        except Exception as e:
            print(f"Error updating comparison plot: {e}")
    
    def _handle_short_recording(self):
        """Handle recording that's too short"""
        if self.formant_calibration_mode:
            word = self.formant_words[self.current_formant_word_index]
            print(f"Recording too short for '{word}', please try again")
            if hasattr(self, 'score_display'):
                self.score_display.set_text(f"Recording too short for '{word}' - try again")
                self.score_display.set_color('red')
        else:
            if hasattr(self, 'score_display'):
                self.score_display.set_text("Recording too short - Please try again")
                self.score_display.set_color('red')
    
    def _handle_processing_error(self):
        """Handle processing errors"""
        if self.formant_calibration_mode:
            word = self.formant_words[self.current_formant_word_index]
            if hasattr(self, 'score_display'):
                self.score_display.set_text(f"Error processing '{word}' - try again")
                self.score_display.set_color('red')
        else:
            if hasattr(self, 'score_display'):
                self.score_display.set_text("PROCESSING ERROR - Please try again")
                self.score_display.set_color('red')
    
    def load_existing_formant_calibration(self):
        """Try to load existing formant calibration"""
        if os.path.exists("formant_calibration.json"):
            if self.formant_calibrator.load_calibration("formant_calibration.json"):
                print("Loaded existing formant calibration")
                
                # Create personalized visualization handlers
                self.vis_realtime = PersonalizedVisualizationHandler(
                    self.sample_rate, self.n_channels, self.formant_calibrator
                )
                self.vis_file = PersonalizedVisualizationHandler(
                    self.sample_rate, self.n_channels, self.formant_calibrator
                )
                
                self.needs_formant_calibration = False
                
                # Show loaded calibration info
                boundaries = self.formant_calibrator.get_formant_boundaries()
                print(f"Your calibrated voice characteristics:")
                print(f"  F1 range: {boundaries['f1_lo']:.0f}-{boundaries['f1_hi']:.0f} Hz")
                print(f"  F2 range: {boundaries['f2_lo']:.0f}-{boundaries['f2_hi']:.0f} Hz")
                
                # Update display
                if hasattr(self, 'score_display'):
                    self.score_display.set_text("Voice calibration loaded - ready to practice")
                    self.score_display.set_color('lime')
                
                return True
        return False
    
    def skip_formant_calibration(self):
        """Skip formant calibration and use default voice mapping"""
        self.needs_formant_calibration = False
        self.formant_calibration_mode = False
        print("Skipping formant calibration - using default voice mapping")
        
        if hasattr(self, 'score_display'):
            self.score_display.set_text("Using default colors - ready to practice")
            self.score_display.set_color('white')
    
    def play_reference_audio(self, event=None):
        """Play the reference audio file"""
        print("Playing reference audio...")
        if hasattr(self, 'audio_data'):
            # Placeholder for audio playback
            print(f"Playing audio file: {self.audio_file_path}")
            # self.audio_playback.play(self.audio_data)
        else:
            print("No reference audio loaded")
    
    def set_reference_text(self, text: str):
        """Set the reference text for comparison"""
        self.file_transcription = text
        print(f"Reference text set to: '{text}'")
    
    def set_pronunciation_guide(self, guide: str):
        """Set pronunciation guide"""
        self.pronunciation_guide = guide
        print(f"Pronunciation guide: {guide}")
    
    def set_translated_text(self, translation: str):
        """Set translation"""
        self.translation = translation
        print(f"Translation: {translation}")
    
    def _load_audio_file(self):
        """Load the audio file for file processing"""
        try:
            print(f"Loading audio file: {self.audio_file_path}")
            self.audio_data, self.original_sr = librosa.load(self.audio_file_path, sr=None)
            
            # Resample if necessary
            if self.original_sr != self.sample_rate:
                self.audio_data = librosa.resample(self.audio_data, 
                                                 orig_sr=self.original_sr, 
                                                 target_sr=self.sample_rate)
                print(f"Resampled from {self.original_sr}Hz to {self.sample_rate}Hz")
            
            print(f"Audio loaded: {len(self.audio_data)} samples, {len(self.audio_data)/self.sample_rate:.2f} seconds")
            
            # Set reference text based on filename or default
            filename = os.path.splitext(os.path.basename(self.audio_file_path))[0]
            self.set_reference_text(filename.lower() if filename else 'hello')
            self.set_pronunciation_guide('heh-low')
            self.set_translated_text('greeting')
            
            # Try to load existing formant calibration
            if not self.load_existing_formant_calibration():
                print("No existing formant calibration found - will calibrate on first recording")
            
            # Process reference audio through SAI
            self._process_reference_audio()
            
        except Exception as e:
            print(f"Error loading audio file: {e}")
    
    def _process_reference_audio(self):
        """Process reference audio through CARFAC and SAI"""
        try:
            print("Processing reference audio...")
            
            # Placeholder processing (uncomment when modules available)
            # ref_carfac = RealCARFACProcessor(fs=self.sample_rate)
            # ref_sai = sai.SAI(self.sai_params)
            
            ref_sai_frames = []
            chunk_size = self.chunk_size
            
            for i in range(0, len(self.audio_data), chunk_size):
                chunk = self.audio_data[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
                
                # nap_output = ref_carfac.process_chunk(chunk)
                # sai_output = ref_sai.RunSegment(nap_output)
                # For now, create dummy SAI output
                sai_output = np.random.random((self.n_channels, self.sai_width)) * np.max(np.abs(chunk))
                ref_sai_frames.append(sai_output)
            
            # Store reference SAI for comparison
            self.reference_sai = np.array(ref_sai_frames)
            
            # Update reference visualization
            self._update_reference_visualization()
            
            print("Reference audio processed")
            
        except Exception as e:
            print(f"Error processing reference audio: {e}")
    
    def _update_reference_visualization(self):
        """Update the reference audio visualization"""
        try:
            if hasattr(self, 'reference_sai') and self.vis_file:
                # Average across frames for display
                avg_sai = np.mean(self.reference_sai, axis=0)
                
                # Convert to color using personalized mapping (or default if not calibrated)
                color_frame = self.vis_file.sai_to_formant_colors(avg_sai)
                
                # Display in reference SAI plot
                self.ax_sai_file.clear()
                self.ax_sai_file.imshow(color_frame, aspect='auto', origin='lower')
                self.ax_sai_file.set_title('Reference Audio SAI', color='white')
                self.ax_sai_file.set_xlabel('Time (samples)')
                self.ax_sai_file.set_ylabel('Frequency Channel')
                
                # Update waveform plot
                self._update_waveform_plot()
                
                plt.draw()
                
        except Exception as e:
            print(f"Error updating reference visualization: {e}")
    
    def _update_waveform_plot(self):
        """Update the waveform and spectrogram plots"""
        try:
            if hasattr(self, 'audio_data'):
                # Waveform
                self.ax_waveform.clear()
                time_axis = np.linspace(0, len(self.audio_data)/self.sample_rate, len(self.audio_data))
                self.ax_waveform.plot(time_axis, self.audio_data, color='cyan', alpha=0.8)
                self.ax_waveform.set_xlabel('Time (s)', color='white')
                self.ax_waveform.set_ylabel('Amplitude', color='white')
                self.ax_waveform.set_title('Reference Audio Waveform', color='white')
                self.ax_waveform.grid(True, alpha=0.3)
                
                # Spectrogram
                self.ax_spectrogram.clear()
                f, t, Sxx = scipy.signal.spectrogram(self.audio_data, self.sample_rate, nperseg=1024)
                self.ax_spectrogram.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
                self.ax_spectrogram.set_xlabel('Time (s)', color='white')
                self.ax_spectrogram.set_ylabel('Frequency (Hz)', color='white')
                self.ax_spectrogram.set_title('Spectrogram', color='white')
                self.ax_spectrogram.set_ylim(0, 4000)  # Focus on speech frequencies
                
        except Exception as e:
            print(f"Error updating waveform plot: {e}")
    
    def start(self):
        """Start the pronunciation trainer"""
        print("Starting Dual SAI Pronunciation Trainer with Personalized Voice Calibration")
        print("=" * 70)
        
        if self.audio_file_path:
            print(f"Reference audio: {self.audio_file_path}")
            print(f"Reference text: '{self.file_transcription}'")
        
        if self.formant_calibrator.is_calibrated:
            print("Voice calibration loaded and ready")
        else:
            print("Click 'Voice Calibration' to calibrate colors to your voice")
            print("Or click 'Skip Calibration' to use default colors")
        
        print("\nInstructions:")
        print("1. First calibrate your voice by saying the 6 words clearly")
        print("2. Then practice pronunciation by clicking 'Record'")
        print("3. Compare your pronunciation with the reference")
        print("4. Colors will be optimized for your vocal tract characteristics")
        
        # Show the plot
        plt.show()


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Dual SAI Pronunciation Trainer with Formant Calibration')
    
    parser.add_argument('audio_file', nargs='?', default=None,
                       help='Path to reference audio file')
    parser.add_argument('--chunk-size', type=int, default=1024,
                       help='Audio chunk size (default: 1024)')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Audio sample rate (default: 16000)')
    parser.add_argument('--sai-width', type=int, default=200,
                       help='SAI width parameter (default: 200)')
    parser.add_argument('--whisper-model', default='base',
                       help='Whisper model size (default: base)')
    parser.add_argument('--language', default='en',
                       help='Language for transcription (default: en)')
    parser.add_argument('--playback-speed', type=float, default=3.0,
                       help='Audio playback speed multiplier (default: 3.0)')
    parser.add_argument('--recording-duration', type=float, default=3.0,
                       help='Recording duration in seconds (default: 3.0)')
    parser.add_argument('--skip-formant-calibration', action='store_true',
                       help='Skip formant calibration phase')
    parser.add_argument('--force-formant-calibration', action='store_true',
                       help='Force new formant calibration even if one exists')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    try:
        # Create the pronunciation trainer
        processor = DualSAIWithRecording(
            audio_file_path=args.audio_file,
            chunk_size=args.chunk_size,
            sample_rate=args.sample_rate,
            sai_width=args.sai_width,
            whisper_model=args.whisper_model,
            debug=args.debug,
            playback_speed=args.playback_speed,
            language=args.language,
        )
        
        # Set recording duration
        processor.recording_duration = args.recording_duration
        
        # Handle calibration options
        if args.skip_formant_calibration:
            processor.skip_formant_calibration()
        elif args.force_formant_calibration:
            processor.needs_formant_calibration = True
            # Delete existing calibration file
            if os.path.exists("formant_calibration.json"):
                os.remove("formant_calibration.json")
                print("Existing calibration deleted - will recalibrate")
        
        # Start the trainer
        processor.start()
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 0
    except Exception as e:
        print(f"Error in pronunciation trainer: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


"""
Real-time calibration workflow:
1. User starts the system
2. System checks for existing formant calibration
3. If none exists, enters formant calibration mode
4. User says 6 words: "bee", "bay", "bad", "law", "bow", "boo"
5. System extracts F1/F2 formants from each word's SAI data
6. Calculates personalized frequency boundaries
7. Creates custom vowel matrix using user's formant ranges
8. Both reference and user audio now use same color mapping
9. Saves calibration for future sessions

Usage Examples:
python dual_sai_with_formant_calibration.py hello.wav
python dual_sai_with_formant_calibration.py --skip-formant-calibration audio.wav
python dual_sai_with_formant_calibration.py --force-formant-calibration --debug audio.wav
"""