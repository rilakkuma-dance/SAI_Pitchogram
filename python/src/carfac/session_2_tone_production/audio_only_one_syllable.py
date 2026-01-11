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
import subprocess

# JAX/CARFAC imports (Optional - kept from original)
try:
    sys.path.append('./jax')
    import jax
    import jax.numpy as jnp
    # import carfac.jax.carfac as carfac
    # from carfac.np.carfac import CarParams
    # import sai
    JAX_AVAILABLE = True
except ImportError:
    print("Warning: JAX/CARFAC/SAI not found. Running in standard mode.")
    JAX_AVAILABLE = False

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

class PracticeSet:
    """Manages practice sets using the provided Tone 1-4 list"""
    
    def __init__(self, audio_base_path="mandarin_audio_one_syllable"):
        self.all_items = []
        self.audio_base_path = Path(audio_base_path)
        
        # --- UPDATED VOCAB LIST ---
        self.vocab_items = [
            # Tone 1
            {"id": 1,  "chinese": "天", "pinyin": "tiān", "tone": "1", "audio": "01_天_1.wav"},
            {"id": 2,  "chinese": "心", "pinyin": "xīn",  "tone": "1", "audio": "02_心_1.wav"},
            {"id": 3,  "chinese": "车", "pinyin": "chē",  "tone": "1", "audio": "03_车_1.wav"},
            # Tone 2
            {"id": 4,  "chinese": "学", "pinyin": "xué",  "tone": "2", "audio": "04_学_2.wav"},
            {"id": 5,  "chinese": "人", "pinyin": "rén",  "tone": "2", "audio": "05_人_2.wav"},
            {"id": 6,  "chinese": "白", "pinyin": "bái",  "tone": "2", "audio": "06_白_2.wav"},
            # Tone 3
            {"id": 7,  "chinese": "老", "pinyin": "lǎo",  "tone": "3", "audio": "07_老_3.wav"},
            {"id": 8,  "chinese": "火", "pinyin": "huǒ",  "tone": "3", "audio": "08_火_3.wav"},
            {"id": 9,  "chinese": "狗", "pinyin": "gǒu",  "tone": "3", "audio": "09_狗_3.wav"},
            # Tone 4
            {"id": 10, "chinese": "叫", "pinyin": "jiào", "tone": "4", "audio": "10_叫_4.wav"},
            {"id": 11, "chinese": "骂", "pinyin": "mà",   "tone": "4", "audio": "11_骂_4.wav"},
            {"id": 12, "chinese": "去", "pinyin": "qù",   "tone": "4", "audio": "12_去_4.wav"},
        ]
        
        # Check which files actually exist
        self.all_items = []
        missing_files = []
        
        for item in self.vocab_items:
            audio_path = self.audio_base_path / item['audio']
            if audio_path.exists():
                self.all_items.append(item)
            else:
                missing_files.append(str(audio_path))
        
        print(f"\nWAV audio files found: {len(self.all_items)} / {len(self.vocab_items)}")
        if missing_files:
            print(f"Missing WAV files ({len(missing_files)}):")
            for f in missing_files:
                print(f"  - {f}")
        
        self.current_set = []
        self.current_index = 0
        self.set_number = 0
    
    def generate_new_set(self):
        """Randomly select 5 words from the list"""
        if len(self.all_items) == 0:
            print("ERROR: No items available to create practice set!")
            return []
        
        # Randomly pick 5 words (or fewer if we don't have 5 files)
        num_to_pick = min(3, len(self.all_items))
        self.current_set = random.sample(self.all_items, num_to_pick)
        
        self.current_index = 0
        self.set_number += 1
        print(f"\n=== Practice Set #{self.set_number} ({num_to_pick} Words) ===")
        for i, item in enumerate(self.current_set, 1):
            print(f"{i}. [Tone {item['tone']}] {item['chinese']} ({item['pinyin']})")
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
        return f"{self.current_index + 1} of {len(self.current_set)}"
    
    def get_audio_path(self, item):
        """Get the full path to the audio file for an item"""
        if 'audio' in item:
            return self.audio_base_path / item['audio']
        return None


class SimpleAudioVisualizerWithSAI:
    """Audio learning system with recording and playback - WAV only"""
    
    def __init__(self, chunk_size=512, sample_rate=16000, save_dir="recordings", audio_ref_dir="reference"):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
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
        
        # PyAudio
        self.p = None
        self.stream = None
        
        self._setup_visualization()
    
    def _setup_visualization(self):
        """Create visualization with practice interface"""
        self.fig = plt.figure(figsize=(12, 8))
        
        # Main display area
        self.ax_main = self.fig.add_subplot(111)
        self.ax_main.axis('off')
        
        # Status text
        self.status_text = self.ax_main.text(
            0.02, 0.02, 'Ready', transform=self.ax_main.transAxes,
            color='lime', fontsize=12, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8)
        )
        
        # Practice item display
        current_item = self.practice_set.get_current_item()
        if current_item:
            # Updated to show Tone instead of English/Type
            item_text = f"[Tone {current_item['tone']}] {current_item['chinese']}\n{current_item['pinyin']}"
        else:
            item_text = "No WAV files found!\n\nCheck 'reference' folder"
        
        self.practice_text = self.ax_main.text(
            0.5, 0.5, item_text, transform=self.ax_main.transAxes,
            color='cyan' if current_item else 'red', fontsize=24, verticalalignment='center',
            horizontalalignment='center', weight='bold',
            bbox=dict(boxstyle='round,pad=1.2', facecolor='black', alpha=0.9, 
                      edgecolor='cyan' if current_item else 'red', linewidth=3)
        )
        
        # Progress indicator
        progress_text = f"{self.practice_set.get_progress()}"
        self.progress_text = self.ax_main.text(
            0.98, 0.98, progress_text, transform=self.ax_main.transAxes,
            color='yellow', fontsize=10, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8)
        )
        
        # Control buttons
        from matplotlib.widgets import Button
        
        self.ax_play_button = plt.axes([0.20, 0.08, 0.10, 0.05])
        self.play_button = Button(self.ax_play_button, 'Play Audio', 
                                  color='lightcyan', hovercolor='cyan')
        self.play_button.on_clicked(self.play_reference_audio)
        
        self.ax_rec_button = plt.axes([0.35, 0.08, 0.12, 0.05])
        self.rec_button = Button(self.ax_rec_button, 'Start Recording', 
                                 color='lightgreen', hovercolor='green')
        self.rec_button.on_clicked(self.toggle_recording)
        
        self.ax_save_button = plt.axes([0.53, 0.08, 0.12, 0.05])
        self.save_button = Button(self.ax_save_button, 'Save Recording', 
                                  color='lightblue', hovercolor='blue')
        self.save_button.on_clicked(self.save_recording)
        
        self.ax_next_button = plt.axes([0.70, 0.08, 0.10, 0.05])
        self.next_button = Button(self.ax_next_button, 'Next Item', 
                                  color='lightyellow', hovercolor='yellow')
        self.next_button.on_clicked(self.next_practice_item)
        
        self.fig.patch.set_facecolor('#1a1a2e')
        self.ax_main.set_facecolor('#16213e')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for recording"""
        try:
            if self.is_recording:
                self.recorded_frames.append(in_data)
            
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
    
    def process_audio(self):
        """Process audio - recording only"""
        print("Audio processing started (recording mode)")
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue
    
    def play_reference_audio(self, event=None):
        """Play the reference WAV audio for the current item"""
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
            self.status_text.set_text(f'WAV file not found')
            self.status_text.set_color('red')
            print(f"ERROR: WAV file not found: {audio_path}")
            return
        
        self.playback_thread = threading.Thread(
            target=self._play_audio_file_wav, 
            args=(audio_path,), 
            daemon=True
        )
        self.playback_thread.start()
    
    def _play_audio_file_wav(self, audio_path):
        """Play a WAV file using PyAudio"""
        self.reference_audio_playing = True
        self.status_text.set_text('Playing reference audio...')
        self.status_text.set_color('cyan')
        print(f"Playing: {audio_path}")
        
        try:
            # Open WAV file
            with wave.open(str(audio_path), 'rb') as wf:
                # Create playback stream
                playback_stream = self.p.open(
                    format=self.p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )
                
                # Play audio
                chunk = 1024
                data = wf.readframes(chunk)
                while data and self.running:
                    playback_stream.write(data)
                    data = wf.readframes(chunk)
                
                # Cleanup
                playback_stream.stop_stream()
                playback_stream.close()
                
            self.status_text.set_text('Ready')
            self.status_text.set_color('lime')
            print(f"Finished playing: {audio_path}")
            
        except Exception as e:
            self.status_text.set_text(f'Error playing audio')
            self.status_text.set_color('red')
            print(f"ERROR playing audio: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.reference_audio_playing = False
    
    def next_practice_item(self, event=None):
        """Move to next practice item"""
        # Stop current reference audio
        self.reference_audio_playing = False
        time.sleep(0.1)  # Brief pause
        
        next_item = self.practice_set.next_item()
        if next_item:
            item_text = f"[Tone {next_item['tone']}] {next_item['chinese']}\n{next_item['pinyin']}"
            self.practice_text.set_text(item_text)
            self.progress_text.set_text(f"{self.practice_set.get_progress()}")
            self.status_text.set_text('Ready')
            self.status_text.set_color('lime')
            
            # Auto-play the new reference audio
            threading.Timer(0.3, self.play_reference_audio).start()
        else:
            # --- TRANSITION LOGIC ---
            self.status_text.set_text('✓ Session complete! Switching...')
            self.status_text.set_color('magenta')
            print(f"Completed Set #{self.practice_set.set_number}. Switching to 2-syllable...")
            
            # Force update of UI before closing
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
            # Wait briefly so user sees the message
            time.sleep(1.0)
            
            # Stop current process and launch next
            self.stop()
            self._launch_next_script()
            plt.close('all')
            sys.exit(0)
    
    def generate_new_set(self, event=None):
        """Generate a new random practice set"""
        if len(self.practice_set.all_items) == 0:
            self.status_text.set_text('No audio files available!')
            self.status_text.set_color('red')
            return
        
        self.practice_set.generate_new_set()
        current_item = self.practice_set.get_current_item()
        if current_item:
            item_text = f"[Tone {current_item['tone']}] {current_item['chinese']}\n{current_item['pinyin']}"
            self.practice_text.set_text(item_text)
            progress_text = f"{self.practice_set.get_progress()}"
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
            filename = f"audio_{item_id}_{timestamp}.wav"
            txt_filename = f"audio_{item_id}_{timestamp}.txt"
        else:
            filename = f"audio_{timestamp}.wav"
            txt_filename = f"audio_{timestamp}.txt"
        
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
                    txt_file.write(f"Chinese: {current_item.get('chinese')}\n")
                    txt_file.write(f"Pinyin: {current_item.get('pinyin')}\n")
                    txt_file.write(f"Tone: {current_item.get('tone')}\n")
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
            return [self.status_text, self.practice_text, self.progress_text]
        except Exception as e:
            print(f"Visualization error: {e}")
            return []
    
    def start(self):
        """Start the learning system"""
        print("Starting Chinese Audio Learning System (WAV Mode)...")
        print(f"Total available items: {len(self.practice_set.all_items)}")
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
        
        animation_interval = 100
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

    def _launch_next_script(self):
        """Finds and launches the two-syllable recognition script"""
        specific_path = Path(r"C:\Users\maruk\carfac-SAI\python\src\carfac\audio_only_two_syllable.py")
        
        script_dir = Path(__file__).parent.resolve()
        filename = "audio_only_two_syllable.py"
        
        possible_locations = [
            specific_path,
            script_dir / filename,
            script_dir.parent / filename,
            Path.cwd() / filename
        ]
        
        next_script = None
        for loc in possible_locations:
            if loc.exists():
                next_script = loc
                break
        
        if next_script:
            print(f"\n🚀 Launching next module: {next_script.name}")
            try:
                subprocess.Popen([sys.executable, str(next_script)])
            except Exception as e:
                print(f"❌ Error launching script: {e}")
        else:
            print(f"\n⚠️ Could not find {filename}")
            print(f"   Checked: {specific_path} and current folders.")

if __name__ == "__main__":
    # Auto-detect the reference directory relative to the script location
    script_dir = Path(__file__).parent  # Directory where this script is located
    audio_ref_dir = script_dir / "mandarin_audio_one_syllable"  # reference folder in same directory as script
    
    print(f"Script location: {script_dir}")
    print(f"Looking for WAV files in: {audio_ref_dir}")
    
    visualizer = SimpleAudioVisualizerWithSAI(
        chunk_size=512,
        sample_rate=16000,
        save_dir="recordings",
        audio_ref_dir=str(audio_ref_dir)
    )
    
    try:
        visualizer.start()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        visualizer.stop()