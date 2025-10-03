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
from pathlib import Path
import time

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

class AudioProcessor:
    """CARFAC audio processor"""
    def __init__(self, fs=16000):
        self.fs = fs
        self.hypers, self.weights, self.state = carfac.design_and_init_carfac(
            carfac.CarfacDesignParameters(fs=fs, n_ears=1)
        )
        self.n_channels = self.hypers.ears[0].car.n_ch
        self.run_segment_jit = jax.jit(carfac.run_segment, static_argnames=['hypers', 'open_loop'])
        print(f"CARFAC initialized with {self.n_channels} channels")

    def process_chunk(self, audio_chunk):
        if len(audio_chunk.shape) == 1:
            audio_input = audio_chunk.reshape(-1, 1)
        else:
            audio_input = audio_chunk
        audio_jax = jnp.array(audio_input, dtype=jnp.float32)
        naps, _, self.state, _, _, _ = self.run_segment_jit(
            audio_jax, self.hypers, self.weights, self.state, open_loop=False
        )
        return np.array(naps[:, :, 0]).T

class SAIProcessor:
    """SAI processor"""
    def __init__(self, sai_params):
        self.sai_params = sai_params
        self.sai = sai.SAI(sai_params)
        print(f"SAI initialized: {sai_params.sai_width} width, {sai_params.num_channels} channels")
    
    def RunSegment(self, nap_output):
        return self.sai.RunSegment(nap_output)

class VisualizationHandler:
    """SAI visualization handler"""
    def __init__(self, sample_rate, sai_params):
        self.sample_rate = sample_rate
        self.sai_params = sai_params
        self.img = np.zeros((sai_params.num_channels, sai_params.sai_width))
        self.sai_frame = np.zeros((sai_params.num_channels, sai_params.sai_width))
    
    def run_frame(self, sai_output):
        self.sai_frame = sai_output
    
    def draw_column(self, column):
        for ch in range(min(self.sai_params.num_channels, len(self.sai_frame))):
            if ch < len(self.sai_frame) and self.sai_frame.shape[1] > 0:
                column[ch] = np.mean(self.sai_frame[ch, :])

class PracticeSet:
    """Manages practice sets - randomly selects 5 from 30 available items"""
    
    def __init__(self, audio_base_path="reference"):
        self.all_items = []
        self.audio_base_path = Path(audio_base_path)
        
        # 15 words - WAV format
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
        
        all_potential_items = words + sentences
        
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
        elif missing_files:
            print(f"Missing {len(missing_files)} WAV files")
        
        if len(self.all_items) == 0:
            print("\nERROR: No WAV audio files found!")
            print(f"Check directory: {self.audio_base_path}")
            print("Expected structure: reference/men/*.wav and reference/women/*.wav")
        
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
        if not self.current_set:
            self.generate_new_set()
        if self.current_index < len(self.current_set):
            return self.current_set[self.current_index]
        return None
    
    def next_item(self):
        self.current_index += 1
        if self.current_index >= len(self.current_set):
            print(f"\n✓ Completed Set #{self.set_number}!")
            return None
        return self.get_current_item()
    
    def get_progress(self):
        if not self.current_set:
            return "No set active"
        return f"Item {self.current_index + 1} of {len(self.current_set)}"
    
    def get_audio_path(self, item):
        if 'audio' in item:
            return self.audio_base_path / item['audio']
        return None


class SimpleAudioVisualizerWithSAI:
    """Audio visualizer with SAI processing and recording"""
    
    def __init__(self, chunk_size=512, sample_rate=16000, sai_width=400, save_dir="recordings", audio_ref_dir="reference"):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sai_width = sai_width
        self.audio_queue = queue.Queue(maxsize=50)
        self.running = False
        
        # Practice set manager
        self.practice_set = PracticeSet(audio_base_path=audio_ref_dir)
        self.practice_set.generate_new_set()
        
        # Audio playback
        self.reference_audio_playing = False
        self.playback_thread = None
        
        # Recording storage
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.recorded_frames = []
        self.is_recording = False
        
        # Waveform buffer
        self.buffer_size = int(sample_rate * 0.5)
        self.waveform_buffer = np.zeros(self.buffer_size)
        self.buffer_index = 0
        
        # CARFAC and SAI setup
        self.processor = AudioProcessor(fs=sample_rate)
        self.n_channels = self.processor.n_channels
        
        self.sai_params = SAIParams(
            num_channels=self.n_channels,
            sai_width=self.sai_width,
            future_lags=self.sai_width - 1,
            num_triggers_per_frame=2,
            trigger_window_width=self.chunk_size + 1,
            input_segment_width=self.chunk_size,
            channel_smoothing_scale=0.5
        )
        
        self.sai_processor = SAIProcessor(self.sai_params)
        self.vis = VisualizationHandler(sample_rate, self.sai_params)
        
        # PyAudio
        self.p = None
        self.stream = None
        
        self._setup_visualization()
    
    def _setup_visualization(self):
        """Create visualization with SAI"""
        self.fig = plt.figure(figsize=(14, 10))
        gs = self.fig.add_gridspec(2, 1, height_ratios=[9, 0.3])
        
        # SAI display
        self.ax_sai = self.fig.add_subplot(gs[0])
        self.im_sai = self.ax_sai.imshow(
            self.vis.img, aspect='auto', origin='upper',
            interpolation='bilinear', extent=[0, self.sai_width, 0, self.n_channels]
        )
        self.ax_sai.set_title('Stabilized Auditory Image (SAI)', color='white', fontsize=14)
        self.ax_sai.axis('off')
        
        # Controls area
        self.ax_controls = self.fig.add_subplot(gs[1])
        self.ax_controls.axis('off')
        
        # Status text
        self.status_text = self.ax_sai.text(
            0.02, 0.02, 'Ready', transform=self.ax_sai.transAxes,
            color='lime', fontsize=12, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8)
        )
        
        # Practice item display
        current_item = self.practice_set.get_current_item()
        if current_item:
            item_text = f"[{current_item['type'].upper()}] {current_item['chinese']}\n{current_item['pinyin']}\n{current_item['english']}"
        else:
            item_text = "No WAV files found!\n\nRun MP3 to WAV converter first"
        
        self.practice_text = self.ax_sai.text(
            0.5, 0.5, item_text, transform=self.ax_sai.transAxes,
            color='cyan' if current_item else 'red', fontsize=16, verticalalignment='center',
            horizontalalignment='center', weight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='black', alpha=0.9, 
                     edgecolor='cyan' if current_item else 'red', linewidth=2)
        )
        
        # Progress indicator
        progress_text = f"Set #{self.practice_set.set_number} | {self.practice_set.get_progress()}"
        self.progress_text = self.ax_sai.text(
            0.98, 0.98, progress_text, transform=self.ax_sai.transAxes,
            color='yellow', fontsize=10, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        
        # Control buttons
        from matplotlib.widgets import Button
        
        self.ax_play_button = plt.axes([0.15, 0.03, 0.08, 0.035])
        self.play_button = Button(self.ax_play_button, 'Play', 
                                  color='lightcyan', hovercolor='cyan')
        self.play_button.on_clicked(self.play_reference_audio)
        
        self.ax_rec_button = plt.axes([0.25, 0.03, 0.10, 0.035])
        self.rec_button = Button(self.ax_rec_button, 'Start Recording', 
                                 color='lightgreen', hovercolor='green')
        self.rec_button.on_clicked(self.toggle_recording)
        
        self.ax_save_button = plt.axes([0.37, 0.03, 0.10, 0.035])
        self.save_button = Button(self.ax_save_button, 'Save Recording', 
                                  color='lightblue', hovercolor='blue')
        self.save_button.on_clicked(self.save_recording)
        
        self.ax_next_button = plt.axes([0.49, 0.03, 0.08, 0.035])
        self.next_button = Button(self.ax_next_button, 'Next Item', 
                                  color='lightyellow', hovercolor='yellow')
        self.next_button.on_clicked(self.next_practice_item)
        
        self.ax_newset_button = plt.axes([0.59, 0.03, 0.10, 0.035])
        self.newset_button = Button(self.ax_newset_button, 'New Set', 
                                    color='lightcoral', hovercolor='red')
        self.newset_button.on_clicked(self.generate_new_set)
        
        self.fig.patch.set_facecolor('black')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.09, hspace=0.3)
    
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
        """Process audio with CARFAC and SAI"""
        print("SAI processing started")
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # CARFAC processing
                nap_output = self.processor.process_chunk(audio_chunk)
                
                # SAI processing
                sai_output = self.sai_processor.RunSegment(nap_output)
                self.vis.run_frame(sai_output)
                
                # Update visualization
                self.vis.img[:, :-1] = self.vis.img[:, 1:]
                self.vis.draw_column(self.vis.img[:, -1])
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue
    
    def play_reference_audio(self, event=None):
        """Play the reference audio for the current item"""
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
            self.status_text.set_text(f'Audio file not found')
            self.status_text.set_color('red')
            print(f"Audio file not found: {audio_path}")
            return
        
        self.playback_thread = threading.Thread(
            target=self._play_audio_file, 
            args=(audio_path,), 
            daemon=True
        )
        self.playback_thread.start()
    
    def _play_audio_file(self, audio_path):
        """Play an audio file using PyAudio"""
        self.reference_audio_playing = True
        self.status_text.set_text('Playing reference audio...')
        self.status_text.set_color('cyan')
        print(f"Playing: {audio_path}")
        
        try:
            with wave.open(str(audio_path), 'rb') as wf:
                playback_stream = self.p.open(
                    format=self.p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )
                
                chunk = 1024
                data = wf.readframes(chunk)
                while data and self.running:
                    playback_stream.write(data)
                    data = wf.readframes(chunk)
                
                playback_stream.stop_stream()
                playback_stream.close()
                
            self.status_text.set_text('Audio playback complete')
            self.status_text.set_color('lime')
            print(f"Finished playing: {audio_path}")
            
        except Exception as e:
            self.status_text.set_text(f'Error playing audio')
            self.status_text.set_color('red')
            print(f"Error playing audio: {e}")
        finally:
            self.reference_audio_playing = False
    
    def next_practice_item(self, event=None):
        """Move to next practice item"""
        if len(self.practice_set.all_items) == 0:
            self.status_text.set_text('No audio files available!')
            self.status_text.set_color('red')
            return
        
        next_item = self.practice_set.next_item()
        if next_item:
            item_text = f"[{next_item['type'].upper()}] {next_item['chinese']}\n{next_item['pinyin']}\n{next_item['english']}"
            self.practice_text.set_text(item_text)
            progress_text = f"Set #{self.practice_set.set_number} | {self.practice_set.get_progress()}"
            self.progress_text.set_text(progress_text)
            self.status_text.set_text('Ready for next item')
            self.status_text.set_color('lime')
            print(f"\nNext item: {next_item['chinese']} ({next_item['pinyin']})")
        else:
            self.status_text.set_text('Set completed! Generate new set')
            self.status_text.set_color('yellow')
    
    def generate_new_set(self, event=None):
        """Generate a new random practice set"""
        if len(self.practice_set.all_items) == 0:
            self.status_text.set_text('No audio files available!')
            self.status_text.set_color('red')
            return
        
        self.practice_set.generate_new_set()
        current_item = self.practice_set.get_current_item()
        if current_item:
            item_text = f"[{current_item['type'].upper()}] {current_item['chinese']}\n{current_item['pinyin']}\n{current_item['english']}"
            self.practice_text.set_text(item_text)
            progress_text = f"Set #{self.practice_set.set_number} | {self.practice_set.get_progress()}"
            self.progress_text.set_text(progress_text)
            self.status_text.set_text('New set generated!')
            self.status_text.set_color('cyan')
    
    def toggle_recording(self, event=None):
        """Toggle recording"""
        if not self.is_recording:
            self.recorded_frames = []
            self.is_recording = True
            self.rec_button.label.set_text('Stop Recording')
            self.rec_button.color = 'red'
            self.rec_button.ax.set_facecolor('red')
            self.status_text.set_text('Recording...')
            self.status_text.set_color('red')
            print("Recording started")
        else:
            self.is_recording = False
            self.rec_button.label.set_text('Start Recording')
            self.rec_button.color = 'lightgreen'
            self.rec_button.ax.set_facecolor('lightgreen')
            duration = len(self.recorded_frames) * self.chunk_size / self.sample_rate
            self.status_text.set_text(f'Stopped ({duration:.1f}s) - Click Save')
            self.status_text.set_color('yellow')
            print(f"Recording stopped - {duration:.1f}s")
    
    def save_recording(self, event=None):
        """Save recording to WAV file with metadata TXT file"""
        if not self.recorded_frames:
            print("No recording to save")
            self.status_text.set_text('No recording to save')
            self.status_text.set_color('orange')
            return
        
        current_item = self.practice_set.get_current_item()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if current_item:
            item_id = current_item.get('id', 'unknown')
            filename = f"recording_{item_id}_{timestamp}.wav"
            txt_filename = f"recording_{item_id}_{timestamp}.txt"
        else:
            filename = f"recording_{timestamp}.wav"
            txt_filename = f"recording_{timestamp}.txt"
        
        filepath = os.path.join(self.save_dir, filename)
        txt_filepath = os.path.join(self.save_dir, txt_filename)
        
        try:
            # Save WAV file
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.recorded_frames))
            
            duration = len(self.recorded_frames) * self.chunk_size / self.sample_rate
            
            # Save metadata TXT file
            with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
                txt_file.write(f"Recording Metadata\n")
                txt_file.write(f"=" * 50 + "\n\n")
                txt_file.write(f"Timestamp: {timestamp}\n")
                txt_file.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                txt_file.write(f"Duration: {duration:.2f} seconds\n")
                txt_file.write(f"Sample Rate: {self.sample_rate} Hz\n")
                txt_file.write(f"Audio File: {filename}\n\n")
                
                if current_item:
                    txt_file.write(f"Practice Item Details\n")
                    txt_file.write(f"-" * 50 + "\n")
                    txt_file.write(f"Item ID: {current_item.get('id')}\n")
                    txt_file.write(f"Type: {current_item.get('type').upper()}\n")
                    txt_file.write(f"Chinese: {current_item.get('chinese')}\n")
                    txt_file.write(f"Pinyin: {current_item.get('pinyin')}\n")
                    txt_file.write(f"English: {current_item.get('english')}\n")
                    txt_file.write(f"Reference Audio: {current_item.get('audio')}\n\n")
                
                txt_file.write(f"Practice Set: #{self.practice_set.set_number}\n")
                txt_file.write(f"Progress: {self.practice_set.get_progress()}\n")
            
            print(f"Saved WAV: {filepath} ({duration:.1f}s)")
            print(f"Saved TXT: {txt_filepath}")
            self.status_text.set_text(f'Saved: {filename} + metadata')
            self.status_text.set_color('lime')
            self.recorded_frames = []
        except Exception as e:
            print(f"Error saving: {e}")
            self.status_text.set_text('Error saving')
            self.status_text.set_color('red')
    
    def update_visualization(self, frame):
        """Update visualization"""
        try:
            # Update SAI
            current_max = np.max(self.vis.img) if self.vis.img.size > 0 else 1
            self.im_sai.set_data(self.vis.img)
            self.im_sai.set_clim(vmin=0, vmax=max(1, min(255, current_max * 1.3)))
            
            return [self.im_sai, self.status_text, self.practice_text, self.progress_text]
        except Exception as e:
            print(f"Visualization error: {e}")
            return [self.im_sai]
    
    def start(self):
        """Start visualizer"""
        print("Starting Chinese Audio Learning System with SAI Visualization...")
        print(f"Total available items: {len(self.practice_set.all_items)}")
        print(f"Practice set composition: 3 words + 2 sentences (5 total)")
        print(f"Audio reference directory: {self.practice_set.audio_base_path}")
        print("Audio format: WAV (no ffmpeg needed)")
        
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
        
        plt.show()
    
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
    # Auto-detect the reference directory relative to the script location
    script_dir = Path(__file__).parent  # Directory where this script is located
    audio_ref_dir = script_dir / "reference"  # reference folder in same directory as script
    
    print(f"Script location: {script_dir}")
    print(f"Looking for WAV files in: {audio_ref_dir}")
    
    visualizer = SimpleAudioVisualizerWithSAI(
        chunk_size=512,
        sample_rate=16000,
        sai_width=400,
        save_dir="recordings",
        audio_ref_dir=str(audio_ref_dir)
    )
    
    try:
        visualizer.start()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        visualizer.stop()