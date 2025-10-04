import sys
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.font_manager as fm
import threading
import queue
import wave
import os
import random
from datetime import datetime
from scipy import signal
from scipy.fft import fft
import time
from pathlib import Path

ref = Path("carfac/reference")

# Check what files actually exist
print("=== WOMEN FOLDER ===")
women_files = sorted([f.name for f in (ref / "women").glob("*.mp3")])
print(f"Total files: {len(women_files)}")
print("First 10:", women_files[:10])

print("\n=== MEN FOLDER ===")
if (ref / "men").exists():
    men_files = sorted([f.name for f in (ref / "men").glob("*.mp3")])
    print(f"Total files: {len(men_files)}")
    print("First 10:", men_files[:10])
else:
    print("men/ folder does NOT exist")

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

# pydub is optional but required to play mp3 / other formats
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except Exception:
    PYDUB_AVAILABLE = False
    # We'll still support WAV using wave module only.

# Configure matplotlib to support Chinese characters
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

class SAIParams:
    """SAI parameters"""
    def __init__(self, num_channels, sai_width, future_lags, num_triggers_per_frame,
                 trigger_window_width, input_segment_width, channel_smoothing_scale):
        self.num_channels = num_channels
        self.sai_width = sai_width
        self.future_lags = future_lags
        self.num_triggers_per_frame = num_triggers_per_frame
        self.trigger_window_width = trigger_window_width
        self.input_segment_width = input_segment_width
        self.channel_smoothing_scale = channel_smoothing_scale

class SpectrogramProcessor:
    """Mel spectrogram processor using STFT"""
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=128, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.window = signal.windows.hann(n_fft)
        self.mel_basis = self._create_mel_filterbank()
    
    def _create_mel_filterbank(self):
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
    
    def process_chunk(self, audio_chunk):
        if len(audio_chunk) < self.n_fft:
            audio_chunk = np.pad(audio_chunk, (0, self.n_fft - len(audio_chunk)))
        windowed = audio_chunk[-self.n_fft:] * self.window
        spectrum = np.abs(fft(windowed)[:self.n_fft // 2 + 1])
        spectrum = 20 * np.log10(spectrum + 1e-10)
        spectrum = self.mel_basis @ spectrum
        return spectrum

class VisualizationHandler:
    def __init__(self, n_freq_bins, spec_width):
        self.n_freq_bins = n_freq_bins
        self.spec_width = spec_width
        self.img = np.zeros((n_freq_bins, spec_width))
        self.ref_img = np.zeros((n_freq_bins, spec_width))

class PracticeSet:
    """Manages practice sets - randomly selects 5 from 30 available items"""

    def __init__(self, audio_base_path="reference"):
        self.all_items = []
        self.audio_base_path = Path(audio_base_path)

        words = [
            {"type": "word", "id": 1, "chinese": "书", "pinyin": "shū", "english": "book", "audio": "men/1_men.wav"},
            {"type": "word", "id": 2, "chinese": "女人", "pinyin": "nǚrén", "english": "woman", "audio": "women/2_women.wav"},
            {"type": "word", "id": 3, "chinese": "雄", "pinyin": "xióng", "english": "male/hero", "audio": "men/3_men.wav"},
            {"type": "word", "id": 4, "chinese": "去", "pinyin": "qù", "english": "to go", "audio": "men/4_men.wav"},
            {"type": "word", "id": 6, "chinese": "喜欢", "pinyin": "xǐhuān", "english": "to like", "audio": "women/6_women.wav"},
            {"type": "word", "id": 7, "chinese": "街道", "pinyin": "jiēdào", "english": "street", "audio": "women/7_women.wav"},
            {"type": "word", "id": 8, "chinese": "熊猫", "pinyin": "xióngmāo", "english": "panda", "audio": "men/8_men.wav"},
            {"type": "word", "id": 9, "chinese": "书店", "pinyin": "shūdiàn", "english": "bookstore", "audio": "women/9_women.wav"},
            {"type": "word", "id": 10, "chinese": "去年", "pinyin": "qùnián", "english": "last year", "audio": "men/10_men.wav"},
            {"type": "word", "id": 11, "chinese": "中午", "pinyin": "zhōngwǔ", "english": "noon", "audio": "women/11_women.wav"},
            {"type": "word", "id": 12, "chinese": "椅子", "pinyin": "yǐzi", "english": "chair", "audio": "men/12_men.wav"},
            {"type": "word", "id": 13, "chinese": "学校", "pinyin": "xuéxiào", "english": "school", "audio": "women/13_women.wav"},
            {"type": "word", "id": 14, "chinese": "医院", "pinyin": "yīyuàn", "english": "hospital", "audio": "men/14_men.wav"},
            {"type": "word", "id": 15, "chinese": "游戏", "pinyin": "yóuxì", "english": "game", "audio": "women/15_women.wav"},
            {"type": "word", "id": 16, "chinese": "她", "pinyin": "tā", "english": "she", "audio": "men/16_men.wav"},
        ]
        
        # 15 sentences - WAV format
        sentences = [
            {"type": "sentence", "id": 5, "chinese": "女人去买书", 
             "pinyin": "Nǚrén qù mǎi shū", "english": "The woman goes to buy books", "audio": "women/5_women.wav"},
            {"type": "sentence", "id": 17, "chinese": "我喜欢吃苹果。", 
             "pinyin": "Wǒ xǐhuān chī píngguǒ.", "english": "I like eating apples", "audio": "men/17_men.wav"},
            {"type": "sentence", "id": 18, "chinese": "他去学校学习汉语。", 
             "pinyin": "Tā qù xuéxiào xuéxí Hànyǔ.", "english": "He goes to school to learn Chinese", "audio": "women/18_women.wav"},
            {"type": "sentence", "id": 19, "chinese": "熊猫在公园里玩。", 
             "pinyin": "Xióngmāo zài gōngyuán lǐ wán.", "english": "The panda plays in the park", "audio": "men/19_men.wav"},
            {"type": "sentence", "id": 20, "chinese": "街道上有很多人。", 
             "pinyin": "Jiēdào shàng yǒu hěnduō rén.", "english": "There are many people on the street", "audio": "women/20_women.wav"},
            {"type": "sentence", "id": 21, "chinese": "医院旁边有一家书店。", 
             "pinyin": "Yīyuàn pángbiān yǒu yī jiā shūdiàn.", "english": "There is a bookstore next to the hospital", "audio": "men/21_men.wav"},
            {"type": "sentence", "id": 22, "chinese": "她是一个聪明的女人。", 
             "pinyin": "Tā shì yí ge cōngmíng de nǚrén.", "english": "She is a smart woman", "audio": "women/22_women.wav"},
            {"type": "sentence", "id": 23, "chinese": "我每天中午吃午饭。", 
             "pinyin": "Wǒ měitiān zhōngwǔ chī wǔfàn.", "english": "I eat lunch every day", "audio": "men/23_men.wav"},
            {"type": "sentence", "id": 24, "chinese": "游戏很有趣。", 
             "pinyin": "Yóuxì hěn yǒuqù.", "english": "The game is interesting", "audio": "women/24_women.wav"},
            {"type": "sentence", "id": 25, "chinese": "请坐在椅子上。", 
             "pinyin": "Qǐng zuò zài yǐzi shàng.", "english": "Please sit on the chair", "audio": "men/25_men.wav"},
            {"type": "sentence", "id": 26, "chinese": "我想去北京旅行。", 
             "pinyin": "Wǒ xiǎng qù Běijīng lǚxíng.", "english": "I want to travel to Beijing", "audio": "women/26_women.wav"},
            {"type": "sentence", "id": 27, "chinese": "学校的老师很好。", 
             "pinyin": "Xuéxiào de lǎoshī hěn hǎo.", "english": "The school's teacher is very good", "audio": "men/27_men.wav"},
            {"type": "sentence", "id": 28, "chinese": "他每天早上跑步。", 
             "pinyin": "Tā měitiān zǎoshang pǎobù.", "english": "He jogs every morning", "audio": "women/28_women.wav"},
            {"type": "sentence", "id": 29, "chinese": "我在家里玩游戏。", 
             "pinyin": "Wǒ zài jiā lǐ wán yóuxì.", "english": "I play games at home", "audio": "men/29_men.wav"},
            {"type": "sentence", "id": 30, "chinese": "她喜欢喝茶。", 
             "pinyin": "Tā xǐhuān hē chá.", "english": "She likes drinking tea", "audio": "women/30_women.wav"},
        ]

        self.all_items = words + sentences
        self.current_set = []
        self.current_index = 0
        self.set_number = 0

    def generate_new_set(self):
        """Randomly select 3 words and 2 sentences - words first, then sentences"""
        if len(self.all_items) == 0:
            print("ERROR: No items available to create practice set!")
            return []
        
        words = [item for item in self.all_items if item['type'] == 'word']
        sentences = [item for item in self.all_items if item['type'] == 'sentence']
        
        num_words = min(3, len(words))
        num_sentences = min(2, len(sentences))
        
        selected_words = random.sample(words, num_words) if num_words > 0 else []
        selected_sentences = random.sample(sentences, num_sentences) if num_sentences > 0 else []
        
        # Shuffle words and sentences separately, then combine with words first
        random.shuffle(selected_words)
        random.shuffle(selected_sentences)
        self.current_set = selected_words + selected_sentences  # Words first, then sentences
        
        self.current_index = 0
        self.set_number += 1
        print(f"\n=== Practice Set #{self.set_number} ({num_words} Words + {num_sentences} Sentences) ===")
        for i, item in enumerate(self.current_set, 1):
            print(f"{i}. [{item['type'].upper()}] {item['chinese']} ({item['pinyin']}) - {item['english']}")
        return self.current_set

    def get_current_item(self):
        """Get the current practice item"""
        if not self.current_set:
            self.generate_new_set()
        if self.current_index < len(self.current_set):
            return self.current_set[self.current_index]
        return None

    def next_item(self):
        """Move to next item in set"""
        self.current_index += 1
        if self.current_index >= len(self.current_set):
            print(f"\n✓ Completed Set #{self.set_number}!")
            return None
        return self.get_current_item()

    def get_progress(self):
        """Get current progress string"""
        if not self.current_set:
            return "No set active"
        return f"Item {self.current_index + 1} of {len(self.current_set)}"

    def get_audio_path(self, item):
        """Get the full path to the audio file for an item"""
        if 'audio' in item:
            return self.audio_base_path / item['audio']
        return None


class SimpleAudioVisualizerWithSAI:
    """Audio visualizer with SAI processing and recording"""

    def __init__(self, chunk_size=512, sample_rate=16000, sai_width=400, save_dir="recordings", audio_ref_dir="carfac/reference"):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sai_width = sai_width
        self.audio_queue = queue.Queue(maxsize=50)
        self.ref_audio_queue = queue.Queue(maxsize=200)  # allow for more ref buffering
        self.running = False

        # Practice set manager
        self.practice_set = PracticeSet(audio_base_path=audio_ref_dir)
        self.practice_set.generate_new_set()

        # Audio playback
        self.reference_audio_playing = False
        self.playback_thread = None
        self.ref_processor_thread = None

        # Recording storage
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.recorded_frames = []
        self.is_recording = False

        # Waveform buffer
        self.buffer_size = int(sample_rate * 0.5)
        self.waveform_buffer = np.zeros(self.buffer_size)
        self.buffer_index = 0

        # Replace lines 256-278 with:
        self.processor = SpectrogramProcessor(sample_rate=sample_rate, n_fft=512, hop_length=128, n_mels=128)
        self.ref_processor = SpectrogramProcessor(sample_rate=sample_rate, n_fft=512, hop_length=128, n_mels=128)
        self.n_channels = self.processor.n_mels
        self.vis = VisualizationHandler(self.n_channels, sai_width)

        # PyAudio
        self.p = None
        self.stream = None

        self._setup_visualization()

    def _setup_visualization(self):
        """Create visualization with side-by-side SAI comparison"""
        self.fig = plt.figure(figsize=(16, 10))
        gs = self.fig.add_gridspec(3, 2, height_ratios=[7, 2, 0.3], width_ratios=[1, 1])

        # Left SAI display (Your Audio) - SWAPPED
        self.ax_sai = self.fig.add_subplot(gs[0, 0])
        self.im_sai = self.ax_sai.imshow(
            self.vis.img, aspect='auto', origin='lower',
            interpolation='bilinear', extent=[self.sai_width, 0, 0, self.n_channels],
            cmap='magma', vmin=-80, vmax=0
        )
        self.ax_sai.set_title('Your Audio (Live/Recording)', color='lime', fontsize=13, weight='bold')
        self.ax_sai.axis('off')
        self.ax_sai.set_ylabel('Frequency Channels', color='white', fontsize=10)
        self.ax_sai.tick_params(colors='white')

        # Right SAI display (Reference Audio) - SWAPPED
        self.ax_ref_sai = self.fig.add_subplot(gs[0, 1])
        self.im_ref_sai = self.ax_ref_sai.imshow(
            self.vis.ref_img, aspect='auto', origin='lower',
            interpolation='bilinear', extent=[self.sai_width, 0, 0, self.n_channels],
            cmap='magma', vmin=-80, vmax=0
        )
        self.ax_ref_sai.set_title('Reference Audio (Native Speaker)', color='cyan', fontsize=13, weight='bold')
        self.ax_ref_sai.set_ylabel('Frequency Channels', color='white', fontsize=10)
        self.ax_ref_sai.tick_params(colors='white')
        self.ax_ref_sai.axis('off')
        
        # ... rest of the method stays the same

        # Practice item display (spans both columns)
        self.ax_practice = self.fig.add_subplot(gs[1, :])
        self.ax_practice.axis('off')

        current_item = self.practice_set.get_current_item()
        item_text = f"[{current_item['type'].upper()}] {current_item['chinese']}\n{current_item['pinyin']}\n{current_item['english']}"
        self.practice_text = self.ax_practice.text(
            0.5, 0.5, item_text, transform=self.ax_practice.transAxes,
            color='cyan', fontsize=18, verticalalignment='center',
            horizontalalignment='center', weight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='black', alpha=0.9, edgecolor='cyan', linewidth=2)
        )

        # Status text (bottom left of practice area)
        self.status_text = self.ax_practice.text(
            0.02, 0.02, 'Ready - Click Play to hear reference', transform=self.ax_practice.transAxes,
            color='lime', fontsize=11, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8)
        )

        # Progress indicator (top right of practice area)
        progress_text = f"Set #{self.practice_set.set_number} | {self.practice_set.get_progress()}"
        self.progress_text = self.ax_practice.text(
            0.98, 0.98, progress_text, transform=self.ax_practice.transAxes,
            color='yellow', fontsize=10, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )

        # Control buttons (bottom, spans both columns)
        from matplotlib.widgets import Button

        self.ax_play_button = plt.axes([0.15, 0.02, 0.08, 0.03])
        self.play_button = Button(self.ax_play_button, '🔊 Play Ref',
                                  color='lightcyan', hovercolor='cyan')
        self.play_button.on_clicked(self.play_reference_audio)

        self.ax_rec_button = plt.axes([0.25, 0.02, 0.10, 0.03])
        self.rec_button = Button(self.ax_rec_button, 'Start Recording',
                                 color='lightgreen', hovercolor='green')
        self.rec_button.on_clicked(self.toggle_recording)

        self.ax_save_button = plt.axes([0.37, 0.02, 0.10, 0.03])
        self.save_button = Button(self.ax_save_button, 'Save Recording',
                                  color='lightblue', hovercolor='blue')
        self.save_button.on_clicked(self.save_recording)

        self.ax_next_button = plt.axes([0.49, 0.02, 0.08, 0.03])
        self.next_button = Button(self.ax_next_button, 'Next Item',
                                  color='lightyellow', hovercolor='yellow')
        self.next_button.on_clicked(self.next_practice_item)

        self.ax_newset_button = plt.axes([0.59, 0.02, 0.10, 0.03])
        self.newset_button = Button(self.ax_newset_button, 'New Set',
                                    color='lightcoral', hovercolor='red')
        self.newset_button.on_clicked(self.generate_new_set)

        self.ax_clear_button = plt.axes([0.71, 0.02, 0.10, 0.03])
        self.clear_button = Button(self.ax_clear_button, 'Clear Ref',
                                   color='lightgray', hovercolor='gray')
        self.clear_button.on_clicked(self.clear_reference)

        self.fig.patch.set_facecolor('#0a0a0a')
        self.ax_practice.set_facecolor('#1a1a2e')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.96, bottom=0.06, hspace=0.15, wspace=0.15)

    def add_to_buffer(self, chunk):
        """Add audio to circular buffer"""
        chunk_size = len(chunk)
        if chunk_size >= self.buffer_size:
            self.waveform_buffer = chunk[-self.buffer_size:].copy()
            self.buffer_index = 0
        else:
            end_idx = self.buffer_index + chunk_size
            if end_idx <= self.buffer_size:
                self.waveform_buffer[self.buffer_index:end_idx] = chunk
                self.buffer_index = end_idx % self.buffer_size
            else:
                first_part = self.buffer_size - self.buffer_index
                self.waveform_buffer[self.buffer_index:] = chunk[:first_part]
                self.waveform_buffer[:chunk_size - first_part] = chunk[first_part:]
                self.buffer_index = chunk_size - first_part

    def get_waveform(self):
        """Get ordered waveform"""
        if self.buffer_index == 0:
            return self.waveform_buffer.copy()
        return np.concatenate([
            self.waveform_buffer[self.buffer_index:],
            self.waveform_buffer[:self.buffer_index]
        ])

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback"""
        try:
            audio_float = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            self.add_to_buffer(audio_float)

            if self.is_recording:
                self.recorded_frames.append(in_data)

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

    def process_audio(self):
        print("Spectrogram processing started")
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                spec_column = self.processor.process_chunk(audio_chunk)
                self.vis.img[:, 1:] = self.vis.img[:, :-1]
                self.vis.img[:, 0] = spec_column
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")

    def process_reference_audio(self):
        print("Reference spectrogram processing started")
        self.vis.ref_img = np.zeros((self.n_channels, self.sai_width))
        audio_buffer = np.array([], dtype=np.float32)
        
        while self.running and self.reference_audio_playing:
            try:
                audio_chunk = self.ref_audio_queue.get(timeout=0.1)
                audio_buffer = np.concatenate([audio_buffer, audio_chunk])
                
                while len(audio_buffer) >= self.chunk_size:
                    chunk_to_process = audio_buffer[:self.chunk_size]
                    audio_buffer = audio_buffer[self.chunk_size:]
                    spec_column = self.ref_processor.process_chunk(chunk_to_process)
                    self.vis.ref_img[:, 1:] = self.vis.ref_img[:, :-1]
                    self.vis.ref_img[:, 0] = spec_column
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Reference processing error: {e}")

    def play_reference_audio(self, event=None):
        """Play the reference audio for the current item (with looping)"""
        if self.reference_audio_playing:
            self.status_text.set_text('Audio already playing...')
            return

        current_item = self.practice_set.get_current_item()
        if not current_item:
            self.status_text.set_text('No item selected')
            self.status_text.set_color('orange')
            return

        audio_path = self.practice_set.get_audio_path(current_item)
        if not audio_path or not audio_path.exists():
            self.status_text.set_text(f'Audio file not found: {audio_path}')
            self.status_text.set_color('orange')
            print(f"Audio file not found: {audio_path}")
            return

        # Start playback in separate thread with looping
        self.playback_thread = threading.Thread(
            target=self._play_audio_file_loop,
            args=(audio_path,),
            daemon=True
        )
        self.playback_thread.start()

    def _play_audio_file_loop(self, audio_path: Path):
        """Play an audio file on loop"""
        if not self.p:
            print("PyAudio not initialized.")
            return

        self.reference_audio_playing = True
        self.status_text.set_text('🔊 Playing reference (looping)...')
        self.status_text.set_color('cyan')
        print(f"Playing (loop): {audio_path}")

        # Start reference processing thread if not already
        if self.ref_processor_thread is None or not self.ref_processor_thread.is_alive():
            self.ref_processor_thread = threading.Thread(target=self.process_reference_audio, daemon=True)
            self.ref_processor_thread.start()

        try:
            suffix = audio_path.suffix.lower()
            
            # Load audio data once
            if suffix == '.wav':
                with wave.open(str(audio_path), 'rb') as wf:
                    channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    framerate = wf.getframerate()
                    audio_data = wf.readframes(wf.getnframes())
            else:
                if not PYDUB_AVAILABLE:
                    raise RuntimeError("Non-WAV reference detected but pydub is not available.")
                seg = AudioSegment.from_file(str(audio_path))
                if seg.channels > 1:
                    seg = seg.set_channels(1)
                if seg.frame_rate != self.sample_rate:
                    seg = seg.set_frame_rate(self.sample_rate)
                seg = seg.set_sample_width(2)
                
                channels = seg.channels
                sampwidth = seg.sample_width
                framerate = seg.frame_rate
                audio_data = seg.raw_data

            p_format = self.p.get_format_from_width(sampwidth)
            playback_stream = self.p.open(
                format=p_format,
                channels=channels,
                rate=framerate,
                output=True
            )

            bytes_per_frame = sampwidth * channels
            frames_per_chunk = self.chunk_size

            # Loop indefinitely until stopped
            while self.running and self.reference_audio_playing:
                offset = 0
                
                # Play through the entire audio file
                while offset < len(audio_data) and self.running and self.reference_audio_playing:
                    chunk_bytes = audio_data[offset: offset + frames_per_chunk * bytes_per_frame]
                    if not chunk_bytes:
                        break
                        
                    playback_stream.write(chunk_bytes)
                    
                    # Process for visualization
                    audio_np = np.frombuffer(chunk_bytes, dtype=np.int16)
                    if channels > 1:
                        audio_np = audio_np.reshape(-1, channels).mean(axis=1)
                    audio_float = audio_np.astype(np.float32) / 32768.0
                    
                    # Resample if needed
                    if framerate != self.sample_rate:
                        num_target = int(len(audio_float) * (self.sample_rate / framerate))
                        if num_target > 0:
                            audio_float = np.interp(
                                np.linspace(0, len(audio_float), num_target, endpoint=False),
                                np.arange(len(audio_float)),
                                audio_float
                            )
                    
                    try:
                        self.ref_audio_queue.put_nowait(audio_float)
                    except queue.Full:
                        try:
                            self.ref_audio_queue.get_nowait()
                            self.ref_audio_queue.put_nowait(audio_float)
                        except queue.Empty:
                            pass
                    
                    offset += len(chunk_bytes)
                
                # Loop restarts here automatically

            playback_stream.stop_stream()
            playback_stream.close()
            
            self.status_text.set_text('Reference stopped')
            self.status_text.set_color('yellow')
            print(f"Stopped looping: {audio_path}")

        except Exception as e:
            err_msg = str(e)
            if 'pydub' in err_msg or 'ffmpeg' in err_msg:
                err_msg = "Error decoding reference file (pydub/ffmpeg may be missing)."
            self.status_text.set_text(f'Error playing audio: {err_msg[:30]}')
            self.status_text.set_color('red')
            print(f"Error playing audio: {e}")
        finally:
            self.reference_audio_playing = False

    def _play_audio_file(self, audio_path: Path):
        """Play an audio file (wav or other formats via pydub) and feed frames into ref queue"""
        if not self.p:
            print("PyAudio not initialized.")
            self.status_text.set_text('Audio device not initialized')
            self.status_text.set_color('red')
            return

        self.reference_audio_playing = True
        self.status_text.set_text('🔊 Playing & visualizing reference...')
        self.status_text.set_color('cyan')
        print(f"Playing: {audio_path}")

        # Start reference processing thread if not already
        if self.ref_processor_thread is None or not self.ref_processor_thread.is_alive():
            self.ref_processor_thread = threading.Thread(target=self.process_reference_audio, daemon=True)
            self.ref_processor_thread.start()

        try:
            suffix = audio_path.suffix.lower()
            # If it's a WAV, try using wave.open for efficiency
            if suffix == '.wav':
                with wave.open(str(audio_path), 'rb') as wf:
                    channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    framerate = wf.getframerate()
                    p_format = self.p.get_format_from_width(sampwidth)

                    playback_stream = self.p.open(
                        format=p_format,
                        channels=channels,
                        rate=framerate,
                        output=True
                    )

                    chunk = self.chunk_size
                    data = wf.readframes(chunk)
                    while data and self.running:
                        playback_stream.write(data)

                        # Convert frames to float32 mono [-1,1] for processing queue
                        # Handle stereo by averaging channels
                        audio_np = np.frombuffer(data, dtype=np.int16)
                        if channels > 1:
                            audio_np = audio_np.reshape(-1, channels).mean(axis=1)
                        audio_float = audio_np.astype(np.float32) / 32768.0

                        # If file sample rate differs from processing rate, resample quickly via numpy (simple)
                        if framerate != self.sample_rate:
                            # naive resample by linear interpolation (not ideal, but keeps things simple)
                            num_target = int(len(audio_float) * (self.sample_rate / framerate))
                            if num_target > 0:
                                audio_float = np.interp(
                                    np.linspace(0, len(audio_float), num_target, endpoint=False),
                                    np.arange(len(audio_float)),
                                    audio_float
                                )

                        # Put into reference queue (drop oldest if full)
                        try:
                            self.ref_audio_queue.put_nowait(audio_float)
                        except queue.Full:
                            try:
                                self.ref_audio_queue.get_nowait()
                                self.ref_audio_queue.put_nowait(audio_float)
                            except queue.Empty:
                                pass

                        data = wf.readframes(chunk)

                    playback_stream.stop_stream()
                    playback_stream.close()

            else:
                # Non-wav file: use pydub if available
                if not PYDUB_AVAILABLE:
                    raise RuntimeError("Non-WAV reference detected but pydub is not available. Install pydub and ffmpeg to play mp3 files.")
                seg = AudioSegment.from_file(str(audio_path))
                # Convert to mono and to target framerate for processing
                if seg.channels > 1:
                    seg = seg.set_channels(1)
                if seg.frame_rate != self.sample_rate:
                    seg = seg.set_frame_rate(self.sample_rate)
                seg = seg.set_sample_width(2)  # 16-bit

                raw_data = seg.raw_data
                channels = seg.channels
                sampwidth = seg.sample_width
                framerate = seg.frame_rate

                p_format = self.p.get_format_from_width(sampwidth)
                playback_stream = self.p.open(
                    format=p_format,
                    channels=channels,
                    rate=framerate,
                    output=True
                )

                bytes_per_frame = sampwidth * channels
                total_frames = len(raw_data) // bytes_per_frame
                # stream in chunk-size frames
                frames_per_chunk = self.chunk_size
                offset = 0
                while offset < len(raw_data) and self.running:
                    chunk_bytes = raw_data[offset: offset + frames_per_chunk * bytes_per_frame]
                    if not chunk_bytes:
                        break
                    playback_stream.write(chunk_bytes)

                    # convert to numpy float32 mono [-1,1]
                    audio_np = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32)
                    # if stereo had been left, we'd average; but we set_channels(1) above
                    audio_float = audio_np / 32768.0

                    try:
                        self.ref_audio_queue.put_nowait(audio_float)
                    except queue.Full:
                        try:
                            self.ref_audio_queue.get_nowait()
                            self.ref_audio_queue.put_nowait(audio_float)
                        except queue.Empty:
                            pass

                    offset += len(chunk_bytes)

                playback_stream.stop_stream()
                playback_stream.close()

            self.status_text.set_text('✓ Reference playback complete')
            self.status_text.set_color('lime')
            print(f"Finished playing: {audio_path}")

        except Exception as e:
            # Friendly error message for pydub missing/ffmpeg missing
            err_msg = str(e)
            if 'pydub' in err_msg or 'ffmpeg' in err_msg:
                err_msg = "Error decoding reference file (pydub/ffmpeg may be missing)."
            self.status_text.set_text(f'Error playing audio: {err_msg[:30]}')
            self.status_text.set_color('red')
            print(f"Error playing audio: {e}")
        finally:
            # Allow processing thread to finish gracefully after queue is drained
            self.reference_audio_playing = False
            # Do not immediately kill ref_processor_thread here; it will exit when reference_audio_playing is False
            # Optionally clear ref queue:
            # while not self.ref_audio_queue.empty(): self.ref_audio_queue.get_nowait()

    def clear_reference(self, event=None):
        """Clear the reference SAI display"""
        self.vis.ref_img = np.zeros((self.n_channels, self.sai_width))
        self.status_text.set_text('Reference cleared')
        self.status_text.set_color('yellow')
        print("Reference SAI cleared")

    def next_practice_item(self, event=None):
        """Move to next practice item"""
        # Stop current reference audio
        self.reference_audio_playing = False
        time.sleep(0.1)  # Brief pause
        
        next_item = self.practice_set.next_item()
        if next_item:
            item_text = f"[{next_item['type'].upper()}] {next_item['chinese']}\n{next_item['pinyin']}\n{next_item['english']}"
            self.practice_text.set_text(item_text)
            self.progress_text.set_text(f"Set #{self.practice_set.set_number} | {self.practice_set.get_progress()}")
            self.status_text.set_text('Ready - Click Play to hear reference')
            self.status_text.set_color('lime')
            
            # Auto-play the new reference audio
            threading.Timer(0.3, self.play_reference_audio).start()
        else:
            self.status_text.set_text('✓ All items in current set completed')
            self.status_text.set_color('yellow')
            print(f"Completed Set #{self.practice_set.set_number}")


    def generate_new_set(self, event=None):
        """Generate a completely new practice set"""
        self.reference_audio_playing = False
        time.sleep(0.1)  # Brief pause
        
        self.practice_set.generate_new_set()
        current_item = self.practice_set.get_current_item()
        item_text = f"[{current_item['type'].upper()}] {current_item['chinese']}\n{current_item['pinyin']}\n{current_item['english']}"
        self.practice_text.set_text(item_text)
        self.progress_text.set_text(f"Set #{self.practice_set.set_number} | {self.practice_set.get_progress()}")
        self.status_text.set_text('Ready - Click Play to hear reference')
        self.status_text.set_color('lime')
        
        # Auto-play the new reference audio
        threading.Timer(0.3, self.play_reference_audio).start()


    def toggle_recording(self, event=None):
        """Start/Stop recording"""
        if not self.is_recording:
            self.recorded_frames = []
            self.is_recording = True
            self.status_text.set_text('● Recording...')
            self.status_text.set_color('red')
            print("Recording started")
        else:
            self.is_recording = False
            self.status_text.set_text('Recording stopped')
            self.status_text.set_color('yellow')
            print("Recording stopped")

    def save_recording(self, event=None):
        """Save the recorded audio to a WAV file with metadata text file"""
        if not self.recorded_frames:
            self.status_text.set_text('No recording to save')
            self.status_text.set_color('orange')
            return

        # Get current practice item info
        current_item = self.practice_set.get_current_item()
        
        # Generate timestamp and filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}.wav"
        txt_filename = f"{timestamp}.txt"
        
        # Save WAV file
        save_path = Path(self.save_dir) / filename
        wf = wave.open(str(save_path), 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.recorded_frames))
        wf.close()
        
        # Save metadata text file
        txt_path = Path(self.save_dir) / txt_filename
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Recording: {filename}\n")
            f.write(f"Item ID: {current_item['id']}\n")
            f.write(f"Type: {current_item['type']}\n")
            f.write(f"Chinese: {current_item['chinese']}\n")
            f.write(f"Pinyin: {current_item['pinyin']}\n")
            f.write(f"English: {current_item['english']}\n")
            f.write(f"Set Number: {self.practice_set.set_number}\n")
            f.write(f"Item Position: {self.practice_set.current_index + 1}/{len(self.practice_set.current_set)}\n")
        
        self.status_text.set_text(f'✓ Saved: {filename} + metadata')
        self.status_text.set_color('lime')
        print(f"Recording saved: {save_path}")
        print(f"Metadata saved: {txt_path}")


    def update_visualization(self, frame):
        try:
            self.im_ref_sai.set_data(self.vis.ref_img)
            self.im_sai.set_data(self.vis.img)
            return [self.im_ref_sai, self.im_sai, self.status_text, self.practice_text, self.progress_text]
        except Exception as e:
            print(f"Visualization error: {e}")
            return [self.im_ref_sai, self.im_sai]

    def start(self):
        """Start the SAI visualizer"""
        print("Starting Chinese Audio Learning System with SAI...")
        print(f"Total available items: {len(self.practice_set.all_items)}")
        print(f"Practice set composition: 3 words + 2 sentences (5 total)")
        print(f"Audio reference directory: {self.practice_set.audio_base_path}")

        self.p = pyaudio.PyAudio()

        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback,
                start=True
            )
            print("Audio stream started")
        except Exception as e:
            print(f"Failed to open audio: {e}")
            return

        self.running = True
        threading.Thread(target=self.process_audio, daemon=True).start()

        animation_interval = max(10, int((self.chunk_size / self.sample_rate) * 1000))
        self.animation = animation.FuncAnimation(
            self.fig, self.update_visualization,
            interval=animation_interval, blit=False, cache_frame_data=False
        )

        # Auto-play reference after a short delay
        threading.Timer(0.5, self.play_reference_audio).start()

        plt.show()

    def _on_first_draw(self, event):
        """Called after first draw - auto-play reference"""
        # Disconnect so this only runs once
        self.fig.canvas.mpl_disconnect(self._on_first_draw)
        # Wait a moment for everything to initialize
        threading.Timer(0.5, self.play_reference_audio).start()

    def stop(self):
        """Stop and cleanup"""
        print("Stopping...")
        self.running = False

        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.p:
                self.p.terminate()
        except:
            pass

        plt.close('all')
        print("Stopped")

if __name__ == "__main__":
    visualizer = SimpleAudioVisualizerWithSAI(
        chunk_size=512,
        sample_rate=16000,
        sai_width=400,
        save_dir="recordings",
        audio_ref_dir="carfac/reference"  # Directory containing men/women folders with MP3s or WAVs
    )

    if not PYDUB_AVAILABLE:
        print("Note: pydub not available. MP3 or other compressed reference files will not play. Install pydub and ffmpeg for that support.")

    try:
        visualizer.start()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        visualizer.stop()