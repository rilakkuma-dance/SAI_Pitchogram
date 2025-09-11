import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb, ListedColormap
import threading
import queue
import time
from typing import Optional

class SimplifiedCARFAC:    
    def __init__(self, fs=22050, n_ch=71):
        self.fs = fs
        self.n_ch = n_ch
        
    def process(self, audio_segment):
        # Simulate cochlear filtering with simple filterbank
        frequencies = np.logspace(np.log10(80), np.log10(8000), self.n_ch)
        nap = np.zeros((self.n_ch, len(audio_segment)))
        
        for i, freq in enumerate(frequencies):
            # Simple bandpass filter simulation
            dt = 1.0 / self.fs
            t = np.arange(len(audio_segment)) * dt
            
            # Create simple resonant filter response
            omega = 2 * np.pi * freq
            decay = np.exp(-omega * 0.01 * t)
            response = np.convolve(audio_segment, decay[:100], mode='same')
            
            # Half-wave rectification and scaling
            nap[i, :] = np.maximum(0, response) * (1 + 0.1 * np.random.randn(len(response)))
            
        return nap

class SimplifiedSAI:    
    def __init__(self, num_channels=71, sai_width=100, trigger_window_width=200, chunk_size=1024):
        self.num_channels = num_channels
        self.sai_width = sai_width
        self.trigger_window_width = trigger_window_width
        self.future_lags = sai_width // 4  # Quarter from future
        self.num_triggers_per_frame = 3
        
        # Buffer for accumulating input - ensure it's large enough for chunk_size
        min_buffer_width = sai_width + int(
            (1 + float(self.num_triggers_per_frame - 1) / 2) * trigger_window_width
        )
        self.buffer_width = max(min_buffer_width, chunk_size * 2)  # At least 2x chunk size
        self.input_buffer = np.zeros((num_channels, self.buffer_width))
        
        # Window function for trigger detection
        self.window = np.sin(np.linspace(
            np.pi / trigger_window_width, np.pi, trigger_window_width
        ))
        
    def process_segment(self, nap_segment):
        # Handle case where input segment is larger than buffer capacity
        segment_width = nap_segment.shape[1]
        
        if segment_width >= self.buffer_width:
            # If input is larger than buffer, just use the most recent part
            self.input_buffer = nap_segment[:, -self.buffer_width:]
        else:
            # Normal case: shift buffer and add new input
            overlap_width = self.buffer_width - segment_width
            if overlap_width > 0:
                self.input_buffer[:, :overlap_width] = self.input_buffer[:, -overlap_width:]
                self.input_buffer[:, overlap_width:] = nap_segment
            else:
                # Edge case: exact fit
                self.input_buffer = nap_segment
        
        # Generate SAI frame
        sai_frame = self._stabilize_segment()
        return sai_frame
        
    def _stabilize_segment(self):
        output_buffer = np.zeros((self.num_channels, self.sai_width))
        
        num_samples = self.input_buffer.shape[1]
        window_hop = self.trigger_window_width // 2
        window_start = (num_samples - self.trigger_window_width) - \
                      (self.num_triggers_per_frame - 1) * window_hop
        
        window_range_start = window_start - self.future_lags
        offset_range_start = 1 + window_start - self.sai_width
        
        if offset_range_start <= 0:
            return output_buffer
            
        for i in range(self.num_channels):
            nap_wave = self.input_buffer[i, :]
            
            for w in range(self.num_triggers_per_frame):
                current_window_offset = w * window_hop
                current_window_start = window_range_start + current_window_offset
                
                if current_window_start < 0 or current_window_start + self.trigger_window_width >= num_samples:
                    continue
                    
                # Find trigger point
                trigger_window = nap_wave[
                    current_window_start:current_window_start + self.trigger_window_width
                ]
                
                if len(trigger_window) == len(self.window):
                    windowed_signal = trigger_window * self.window
                    peak_val = np.max(windowed_signal)
                    trigger_time = np.argmax(windowed_signal) + current_window_offset
                    
                    if peak_val <= 0:
                        peak_val = np.max(self.window)
                        trigger_time = np.argmax(self.window) + current_window_offset
                    
                    # Blend segment into output
                    alpha = (0.025 + peak_val) / (0.5 + peak_val)
                    
                    start_idx = trigger_time + offset_range_start
                    end_idx = start_idx + self.sai_width
                    
                    if start_idx >= 0 and end_idx <= num_samples:
                        output_buffer[i, :] *= (1 - alpha)
                        output_buffer[i, :] += alpha * nap_wave[start_idx:end_idx]
        
        return output_buffer

class RealTimeSAIAnimator:    
    def __init__(self, 
                 chunk_size=1024, 
                 sample_rate=22050, 
                 n_channels=71,
                 sai_width=100,
                 update_interval=50):
        
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.sai_width = sai_width
        self.update_interval = update_interval
        
        # Audio setup
        self.audio_queue = queue.Queue(maxsize=10)
        self.running = False
        
        # CARFAC and SAI setup
        self.carfac = SimplifiedCARFAC(fs=sample_rate, n_ch=n_channels)
        self.sai = SimplifiedSAI(
            num_channels=n_channels, 
            sai_width=sai_width,
            trigger_window_width=min(400, chunk_size),
            chunk_size=chunk_size
        )
        
        # Initialize matplotlib components
        self._setup_visualization()
        
        # Audio stream initialization (will be set in start())
        self.p = None
        self.stream = None
    
    def _setup_visualization(self):
        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.sai_data = np.zeros((self.n_channels, self.sai_width))
        
        # Create rainbow HSV colormap
        cmap = self._create_rainbow_colormap()
        
        # Create the image plot
        self.im = self.ax.imshow(
            self.sai_data, 
            aspect='auto', 
            origin='lower',
            cmap=cmap,
            interpolation='nearest',
            vmin=0,
            vmax=1
        )

        plt.tight_layout()
    
    def _create_rainbow_colormap(self):
        n_bins = 256
        
        # Rainbow: Full spectrum from blue to red
        hues = np.linspace(0.7, 0.0, n_bins)     # Blue to red
        saturations = np.ones(n_bins) * 0.9       # High saturation throughout
        values = np.linspace(0.2, 1.0, n_bins)   # Dark to bright
        
        hsv_colors = np.stack([hues, saturations, values], axis=-1)
        rgb_colors = hsv_to_rgb(hsv_colors.reshape(-1, 1, 3)).reshape(-1, 3)
        
        return ListedColormap(rgb_colors)
    
    def audio_callback(self, in_data, frame_count, time_info, status):
                    
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            # Skip if queue is full (processing can't keep up)
            pass
            
        return (in_data, pyaudio.paContinue)
    
    def process_audio(self):
        while self.running:
            try:
                # Get audio data with timeout
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Process through CARFAC
                nap = self.carfac.process(audio_chunk)
                
                # Process through SAI
                sai_frame = self.sai.process_segment(nap)
                
                # Update visualization data
                self.sai_data = sai_frame
                
            except queue.Empty:
                continue
    
    def update_plot(self, frame):
        if self.sai_data is not None:
            # Apply some smoothing and normalization
            smoothed_data = self.sai_data.copy()
            
            # Normalize to [0, 1] range with some dynamic range compression
            max_val = np.percentile(smoothed_data, 95)
            if max_val > 0:
                smoothed_data = np.clip(smoothed_data / max_val, 0, 1)
            
            # Apply gamma correction for better visualization
            smoothed_data = np.power(smoothed_data, 0.7)
            
            self.im.set_data(smoothed_data)
    
    def start(self):      
        try:
            # Initialize PyAudio
            self.p = pyaudio.PyAudio()
            
            # Start audio stream
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback,
                start=False
            )
            
            self.running = True
            
            # Start audio processing thread
            self.audio_thread = threading.Thread(target=self.process_audio)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            # Start audio stream
            self.stream.start_stream()
            
            # Start animation (without blit for stability)
            self.ani = animation.FuncAnimation(
                self.fig, 
                self.update_plot, 
                interval=self.update_interval,
                cache_frame_data=False
            )
            
            plt.show()
            
        except Exception as e:
            self.stop()
    
    def stop(self):        
        self.running = False
        
        plt.close('all')

def main():
    
    try:
        # Create and start the animator
        animator = RealTimeSAIAnimator(
            chunk_size=1024,
            sample_rate=22050,
            n_channels=71,
            sai_width=150,
            update_interval=50  # 20 FPS
        )
        
        animator.start()
        
    finally:
        if 'animator' in locals():
            animator.stop()

if __name__ == "__main__":
    main()