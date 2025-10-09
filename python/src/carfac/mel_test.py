import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft
import wave
from pathlib import Path
import argparse

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not available. Only WAV files supported.")

class MelSpectrogramGenerator:
    """Generate Mel spectrogram from audio file"""
    
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=128, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.window = signal.windows.hann(n_fft)
        self.mel_basis = self._create_mel_filterbank()
    
    def _create_mel_filterbank(self):
        """Create Mel filterbank"""
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        fmin, fmax = 0, self.sample_rate / 2
        mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        filterbank = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for i in range(self.n_mels):
            left, center, right = bin_points[i:i+3]
            for j in range(left, center):
                filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                filterbank[i, j] = (right - j) / (right - center)
        return filterbank
    
    def load_audio(self, audio_path):
        """Load audio file (WAV or other formats via pydub)"""
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Loading: {audio_path}")
        
        # Try pydub first for all formats (more reliable for MP3)
        if PYDUB_AVAILABLE:
            try:
                print("  Using pydub for audio loading...")
                seg = AudioSegment.from_file(str(audio_path))
                
                # Convert to mono
                if seg.channels > 1:
                    print(f"  Converting from {seg.channels} channels to mono")
                    seg = seg.set_channels(1)
                
                # Convert to target sample rate
                if seg.frame_rate != self.sample_rate:
                    print(f"  Resampling from {seg.frame_rate} Hz to {self.sample_rate} Hz")
                    seg = seg.set_frame_rate(self.sample_rate)
                
                # Convert to 16-bit
                seg = seg.set_sample_width(2)
                
                # Get raw data
                raw_data = seg.raw_data
                audio_np = np.frombuffer(raw_data, dtype=np.int16)
                audio_float = audio_np.astype(np.float32) / 32768.0
                
                print(f"  Channels: 1 (mono)")
                print(f"  Sample rate: {self.sample_rate} Hz")
                print(f"  Duration: {len(audio_float) / self.sample_rate:.2f} seconds")
                
                return audio_float
                
            except Exception as e:
                print(f"  pydub failed: {e}")
                if audio_path.suffix.lower() != '.wav':
                    raise RuntimeError(f"Failed to load {audio_path.suffix} file. Make sure ffmpeg is installed.")
                print("  Falling back to wave module for WAV file...")
        
        # Fallback to wave module for WAV files
        if audio_path.suffix.lower() == '.wav':
            try:
                with wave.open(str(audio_path), 'rb') as wf:
                    channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    framerate = wf.getframerate()
                    n_frames = wf.getnframes()
                    audio_data = wf.readframes(n_frames)
                    
                    # Convert to numpy array
                    if sampwidth == 2:  # 16-bit
                        audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    elif sampwidth == 1:  # 8-bit
                        audio_np = np.frombuffer(audio_data, dtype=np.uint8).astype(np.int16)
                        audio_np = (audio_np - 128) * 256  # Convert to 16-bit range
                    else:
                        raise ValueError(f"Unsupported sample width: {sampwidth}")
                    
                    # Convert to mono if stereo
                    if channels > 1:
                        print(f"  Converting from {channels} channels to mono")
                        audio_np = audio_np.reshape(-1, channels).mean(axis=1).astype(np.int16)
                    
                    # Normalize to [-1, 1]
                    audio_float = audio_np.astype(np.float32) / 32768.0
                    
                    print(f"  Channels: {channels} → 1 (mono)")
                    print(f"  Sample rate: {framerate} Hz")
                    print(f"  Duration: {len(audio_float) / framerate:.2f} seconds")
                    
                    # Resample if needed
                    if framerate != self.sample_rate:
                        print(f"  Resampling from {framerate} Hz to {self.sample_rate} Hz")
                        num_target = int(len(audio_float) * (self.sample_rate / framerate))
                        audio_float = np.interp(
                            np.linspace(0, len(audio_float), num_target, endpoint=False),
                            np.arange(len(audio_float)),
                            audio_float
                        )
                    
                    return audio_float
                    
            except Exception as e:
                raise RuntimeError(f"Failed to load WAV file: {e}")
        
        # If we get here, file format not supported
        raise RuntimeError(f"Unsupported file format: {audio_path.suffix}. Install pydub and ffmpeg for MP3 support.")
    
    def generate_mel_spectrogram(self, audio_data):
        """Generate Mel spectrogram from audio data"""
        print("Generating Mel spectrogram...")
        
        # Calculate number of frames
        n_frames = 1 + (len(audio_data) - self.n_fft) // self.hop_length
        
        # Initialize spectrogram
        mel_spec = np.zeros((self.n_mels, n_frames))
        
        # Process each frame
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.n_fft
            
            if end > len(audio_data):
                # Pad last frame if needed
                frame = np.pad(audio_data[start:], (0, end - len(audio_data)))
            else:
                frame = audio_data[start:end]
            
            # Apply window
            windowed = frame * self.window
            
            # FFT
            spectrum = np.abs(fft(windowed)[:self.n_fft // 2 + 1])
            
            # Convert to dB
            spectrum_db = 20 * np.log10(spectrum + 1e-10)
            
            # Apply Mel filterbank
            mel_spec[:, i] = self.mel_basis @ spectrum_db
        
        print(f"  Spectrogram shape: {mel_spec.shape}")
        print(f"  Frequency bins: {mel_spec.shape[0]}")
        print(f"  Time frames: {mel_spec.shape[1]}")
        
        return mel_spec
    
    def plot_mel_spectrogram(self, mel_spec, audio_path, save_path=None):
        """Plot and optionally save Mel spectrogram"""
        duration = mel_spec.shape[1] * self.hop_length / self.sample_rate
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        im = ax.imshow(
            mel_spec,
            aspect='auto',
            origin='lower',
            cmap='magma',
            extent=[0, duration, 0, self.n_mels],
            interpolation='bilinear'
        )
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Mel Frequency Bin', fontsize=12)
        ax.set_title(f'Mel Spectrogram: {Path(audio_path).name}', fontsize=14, weight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Magnitude (dB)', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved plot to: {save_path}")
        
        plt.show()
    
    def process_file(self, audio_path, save_plot=None):
        """Complete pipeline: load audio → generate spectrogram → plot"""
        print(f"\n{'='*60}")
        print(f"MEL SPECTROGRAM GENERATOR")
        print(f"{'='*60}\n")
        
        # Load audio
        audio_data = self.load_audio(audio_path)
        
        # Generate spectrogram
        mel_spec = self.generate_mel_spectrogram(audio_data)
        
        # Plot
        self.plot_mel_spectrogram(mel_spec, audio_path, save_plot)
        
        return mel_spec


def main():
    parser = argparse.ArgumentParser(description='Generate Mel Spectrogram from Audio File')
    parser.add_argument('audio_file', type=str, nargs='?', default=None, help='Path to audio file (WAV, MP3, etc.)')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Target sample rate (default: 16000)')
    parser.add_argument('--n-fft', type=int, default=512, help='FFT size (default: 512)')
    parser.add_argument('--hop-length', type=int, default=128, help='Hop length (default: 128)')
    parser.add_argument('--n-mels', type=int, default=128, help='Number of Mel bands (default: 128)')
    parser.add_argument('--save', type=str, default=None, help='Save plot to file (e.g., spectrogram.png)')
    
    args = parser.parse_args()
    
    # If no audio file provided, use default or prompt
    if args.audio_file is None:
        # Default file path (edit this for your convenience)
        default_path = r"C:\Users\maruk\carfac-SAI\python\src\carfac\reference\a4_FV1_MP3.wav"
        
        if Path(default_path).exists():
            print(f"No audio file specified. Using default: {default_path}")
            args.audio_file = default_path
        else:
            print("Error: No audio file specified")
            print("\nUsage:")
            print('  python mel_test.py "path/to/audio.wav"')
            print('  python mel_test.py "C:\\path\\to\\audio.mp3" --save output.png')
            return 1
    
    # Create generator
    generator = MelSpectrogramGenerator(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels
    )
    
    # Process file
    try:
        mel_spec = generator.process_file(args.audio_file, save_plot=args.save)
        print(f"\n{'='*60}")
        print(f"✓ Successfully generated Mel spectrogram!")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())