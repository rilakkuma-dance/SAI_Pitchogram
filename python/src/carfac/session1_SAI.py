import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, TextBox
import sys
import numpy as np
from pathlib import Path
import sounddevice as sd
import librosa
import random
import os
import time
import subprocess 
import csv 
from datetime import datetime

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # 数式フォントもTimes系に合わせる

# ==========================================
# IMPORT THE NEW CONFIGURATION HELPER
# ==========================================
from sai_config import get_sai_params
# ==========================================

# ==========================================
# FIX: CONFIGURE FONTS FOR CHINESE SUPPORT
# ==========================================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

# JAX/CARFAC/SAI imports
try:
    sys.path.append('./jax')
    import jax
    import jax.numpy as jnp
    import carfac.jax.carfac as carfac
    import sai
    JAX_AVAILABLE = True
except ImportError:
    print("Warning: JAX/CARFAC/SAI not found. Install required packages.")
    JAX_AVAILABLE = False
    sys.exit(1)

from modules.visualization_handler import VisualizationHandler

# -----------------------------------------------------------
# Audio Processor Class
# -----------------------------------------------------------
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
            except Exception:
                self.use_carfac = False
                self.n_channels = 200
        else:
            self.use_carfac = False
            self.n_channels = 200
        print(f"Total Channels: {self.n_channels}")

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
            except Exception:
                pass
        
        # Fallback
        if isinstance(audio_chunk, np.ndarray):
            chunk = audio_chunk.flatten()
        else:
            chunk = np.array(audio_chunk).flatten()
        if chunk.size == 0:
            return np.zeros((self.n_channels, 0), dtype=np.float32)
        abs_chunk = np.abs(chunk)
        nap = np.tile(abs_chunk, (self.n_channels, 1)).astype(np.float32)
        channel_scales = np.linspace(1.0, 0.1, num=self.n_channels, dtype=np.float32)[:, None]
        return nap * channel_scales

# -----------------------------------------------------------
# SAI Processor Class
# -----------------------------------------------------------
class SAIProcessor:
    def __init__(self, sai_params):
        self.sai_params = sai_params
        if JAX_AVAILABLE:
            try:
                self.sai = sai.SAI(sai_params)
                self.use_sai = True
            except Exception:
                self.use_sai = False
        else:
            self.use_sai = False
    
    def RunSegment(self, nap_output):
        if self.use_sai:
            try:
                return self.sai.RunSegment(nap_output)
            except Exception:
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

# -----------------------------------------------------------
# MAIN QUIZ CLASS
# -----------------------------------------------------------
class ToneIntroductionQuizMixed:
    def __init__(self):
        self.script_dir = Path(__file__).parent.resolve()
        self.sample_rate = 16000
        self.chunk_size = 450
        
        # 1. Setup Folders
        self.folder_one = self._find_folder('mandarin_audio_two_syllable')
        self.folder_two = self._find_folder('mandarin_audio_two_syllable')
        
        # 2. Load Vocab (3 from One-Syllable, 3 from Two-Syllable)
        items_one = []
        items_two = []
        
        if self.folder_one:
            items_one = self._scan_folder(self.folder_one, syllable_count=1)
        if self.folder_two:
            items_two = self._scan_folder(self.folder_two, syllable_count=2)
            
        random.shuffle(items_one)
        random.shuffle(items_two)
        
        selected_one = items_one[:3]
        selected_two = items_two[:3]
        
        self.vocab_items = selected_one + selected_two
        random.shuffle(self.vocab_items)
            
        print(f"Loaded {len(self.vocab_items)} files ({len(selected_one)} from folder 1, {len(selected_two)} from folder 2).")
        
        # 3. Control Flags
        self.is_playing = False
        
        # 4. SAI Setup
        self.processor = AudioProcessor(fs=self.sample_rate)
        self.n_channels = self.processor.n_channels
        
        # ========================================================
        # REFACTORED: USING CONFIG FILE
        # ========================================================
        self.sai_params = get_sai_params(self.n_channels, self.chunk_size)
        # ========================================================

        self.sai_processor = SAIProcessor(self.sai_params)
        self.vis = VisualizationHandler(self.sample_rate, self.sai_params)
        
        # RGB Buffer (Note: Width 400 must match SAI_WIDTH in sai_config.py)
        # Ideally, you should import SAI_WIDTH from config too, but hardcoding 400 is okay for now.
        self.rgb_img = np.zeros((self.n_channels, 400, 3), dtype=np.float32)

        # 5. Audio Playback Variables
        self.audio_data = None
        self.current_frame_index = 0
        
        # 6. Quiz State
        self.current_item = None
        self.answered = False
        self.question_count = 0
        self.max_questions = len(self.vocab_items)
        self.results = [] 
        
        self.fig = plt.figure(figsize=(10, 10))
        self.fig.patch.set_facecolor('white')
        
        self.timer_started = False
        self.question_start_time = None
        
        self._setup_interface()
        
        if self.vocab_items:
            self._select_next_item()
        else:
            print("ERROR: No audio files found.")

    def _find_folder(self, folder_name):
        path = self.script_dir / folder_name
        if path.exists(): return path
        path = self.script_dir.parent / folder_name
        if path.exists(): return path
        return None

    def _scan_folder(self, folder_path, syllable_count):
        items = []
        if not folder_path: return items
        
        for file_path in folder_path.glob('*.wav'):
            try:
                parts = file_path.stem.split('_')
                if len(parts) >= 3:
                    tone = parts[-1]
                    chinese = parts[-2]
                    
                    items.append({
                        "id": file_path.name,
                        "chinese": chinese,
                        "tone": tone,
                        "audio_path": file_path,
                        "syllables": syllable_count
                    })
            except Exception as e:
                print(f"Skipping {file_path.name}")
        return items

    def _setup_interface(self):
        self.ax_ui = self.fig.add_axes([0, 0, 1, 1])
        self.ax_ui.axis('off')

        self.ax_sai = self.fig.add_axes([0.12, 0.58, 0.76, 0.32])
        self.ax_sai.axis('off')
        
        self.im_sai = self.ax_sai.imshow(
            self.rgb_img,
            aspect='auto', origin='upper', extent=[0, 11.25,self.processor.n_channels, 0]
        )
        # 1. Progress
        self.progress_text = self.ax_ui.text(0.5, 0.50, '', 
                                             ha='center', fontsize=12, color='#7f8c8d')
        
        # 2. Status
        self.status_text = self.ax_ui.text(0.5, 0.46, 'Click Play Loop to start', 
                                           ha='center', fontsize=10, color='#7f8c8d')

        self.instructions = self.ax_ui.text(0.5, 0.42, 
            "Each audio contains one or two tones. Identify the tone(s) and enter the corresponding number(s) (e.g., 1, 2, 12, 31).",
            ha='center', va='top', fontsize=9, color='black')

        self.ax_ui.text(0.28, 0.33, 'Tone(s):', ha='right', va='center', fontsize=10)
        ax_input = plt.axes([0.3, 0.30, 0.4, 0.06]) 
        self.text_input = TextBox(ax_input, '', color='white', hovercolor='#f9f9f9')

        self.ax_ui.text(0.35, 0.25, 'Your answer:', ha='right', fontsize=10, color='#7f8c8d')
        self.answer_text = self.ax_ui.text(0.36, 0.25, '', ha='left', fontsize=10, weight='bold')
        self.feedback_text = self.ax_ui.text(0.65, 0.25, 'Feedback', ha='left', fontsize=10, color='#7f8c8d')

        self.ax_btn = plt.axes([0.3, 0.12, 0.4, 0.08])
        self.btn_action = Button(self.ax_btn, 'Play Loop', color='#3498db', hovercolor='#3498db')
        self.btn_action.label.set_color('white')
        self.btn_action.label.set_weight('bold')
        self.btn_action.label.set_fontsize(14)
        self.btn_action.on_clicked(self._handle_button_click)

    def _start_loop(self):
        if not self.current_item: return
        audio_path = self.current_item['audio_path']
        if audio_path.exists():
            try:
                raw_audio, _ = librosa.load(str(audio_path), sr=self.sample_rate)
                self.audio_data, _ = librosa.effects.trim(raw_audio, top_db=25)
                self.audio_data = np.pad(self.audio_data, (0, 3000), 'constant') 
                self.current_frame_index = 0
                self.is_playing = True
                sd.play(self.audio_data, self.sample_rate)
                
                if not self.timer_started:
                    self.question_start_time = time.time()
                    self.timer_started = True

                self.btn_action.label.set_text('Check Answer')
                self.btn_action.ax.set_facecolor('#3498db') 
                
                # Hide info
                self.status_text.set_text(f'Playing: ???') 
                self.status_text.set_color('#3498db')
            except Exception: pass

    def _handle_button_click(self, event):
        if not self.is_playing and not self.answered:
            self._start_loop()
        elif not self.answered:
            self.check_answer(self.text_input.text) 
        else:
            self._next_word()

    def check_answer(self, text):
        if not self.current_item or self.answered: return
        user_answer = text.strip().replace(' ', '').replace(',', '').replace('-', '')
        correct_answer = self.current_item['tone'].replace(',', '').replace('-', '')
        
        # Calculate time
        time_taken = 0.0
        if self.question_start_time:
            time_taken = time.time() - self.question_start_time
        
        self.answered = True
        is_correct = (user_answer == correct_answer)
        self.answer_text.set_text(user_answer)
        
        # --- SAVE RESULT TO MEMORY ---
        self.results.append({
            'question_idx': self.question_count + 1,
            'chinese': self.current_item['chinese'],
            'syllables': self.current_item['syllables'],
            'correct_tone': correct_answer,
            'user_answer': user_answer,
            'is_correct': is_correct,
            'time_seconds': round(time_taken, 2),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        # -----------------------------
        
        # Reveal Info
        self.status_text.set_text(f'Revealed: {self.current_item["chinese"]} (Tone {correct_answer})')
        self.status_text.set_color('#555555')
        
        if is_correct:
            self.feedback_text.set_text('CORRECT!')
            self.feedback_text.set_color('#27ae60')
        else:
            self.feedback_text.set_text(f'INCORRECT (Correct: {correct_answer})')
            self.feedback_text.set_color('#e74c3c')
            
        self.btn_action.label.set_text('Next Question')
        self.btn_action.ax.set_facecolor('#27ae60')

    def _next_word(self):
        self.question_count += 1
        # Check if done
        if self.question_count >= self.max_questions or self.question_count >= len(self.vocab_items):
            print("Quiz Completed.")
            self.is_playing = False
            sd.stop()
            
            # --- SAVE TO CSV ---
            self._save_results_to_file()
            # -------------------
            
            plt.close(self.fig)
        else:
            self._select_next_item()

    def _save_results_to_file(self):
        # 1. Use a fixed filename
        filename = "session1_SAI_results.csv"
        filepath = self.script_dir / filename
        
        # 2. Check if file exists so we only write headers once
        file_exists = filepath.exists()
        
        try:
            # 3. Use mode='a' (Append) instead of 'w' (Write)
            with open(filepath, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=[
                    'question_idx', 'chinese', 'syllables', 'correct_tone', 
                    'user_answer', 'is_correct', 'time_seconds', 'timestamp'
                ])
                
                # Only write the top header row if the file is brand new
                if not file_exists:
                    writer.writeheader()
                
                writer.writerows(self.results)
                
            print(f"Results appended to {filepath}")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    def _select_next_item(self):
        self.is_playing = False
        self.answered = False
        self.timer_started = False
        self.question_start_time = None
        sd.stop()
        self.rgb_img[:] = 0
        self.vis.img[:] = 0
        self.im_sai.set_data(self.rgb_img)
        self.text_input.set_val('')
        self.answer_text.set_text('')
        self.feedback_text.set_text('Feedback')
        self.current_item = self.vocab_items[self.question_count]
        self.btn_action.label.set_text('Play')
        self.btn_action.ax.set_facecolor('#5B5FED')
        self._update_progress()

    def _update_progress(self):
        self.progress_text.set_text(f"")

    def update_animation(self, frame):
        if not self.is_playing or self.audio_data is None:
            return [self.im_sai]

        if self.current_frame_index + self.chunk_size < len(self.audio_data):
            chunk = self.audio_data[self.current_frame_index : self.current_frame_index + self.chunk_size]
            self.current_frame_index += self.chunk_size
        else:
            self.current_frame_index = 0
            chunk = self.audio_data[0 : self.chunk_size]
            try: sd.play(self.audio_data, self.sample_rate)
            except: pass

        nap_output = self.processor.process_chunk(chunk)
        sai_output = self.sai_processor.RunSegment(nap_output)
        self.vis.get_vowel_embedding(nap_output)
        self.vis.run_frame(sai_output)
        
        if self.vis.img.shape[1] > 1:
            self.vis.img[:, :-1] = self.vis.img[:, 1:]
            self.vis.draw_column(self.vis.img[:, -1])

        # Vowel Color + Brightness Boost
        vowel_coords = getattr(self.vis, 'vowel_coords', np.array([0.0, 0.0])).flatten()
        vc_x = float(vowel_coords[0]) if len(vowel_coords) > 0 else 0.0
        vc_y = float(vowel_coords[1]) if len(vowel_coords) > 1 else 0.0
        
        r_val = 0.5 - 0.6 * vc_y
        g_val = 0.5 - 0.6 * vc_x
        b_val = 0.35 * (vc_x + vc_y) + 0.4
        tint = np.clip([r_val, g_val, b_val], 0.0, 1.0)
        
        if self.vis.img.ndim == 3: brightness_col = np.mean(self.vis.img[:, -1, :], axis=1)
        else: brightness_col = self.vis.img[:, -1]
             
        target_height = self.rgb_img.shape[0]
        source_height = brightness_col.shape[0]
        if source_height != target_height:
            norm_col = np.interp(np.linspace(0, 1, target_height), np.linspace(0, 1, source_height), brightness_col)
        else: norm_col = brightness_col

        current_max = np.max(self.vis.img) if self.vis.img.size else 1.0
        if current_max < 1e-6: current_max = 1.0
        norm_col = np.clip(norm_col / (current_max * 0.8), 0, 1)
        
        # Apply tint and BOOST
        colored_col = (norm_col[:, None] * tint[None, :]) * 2.5 
        
        self.rgb_img[:, :-1, :] = self.rgb_img[:, 1:, :]
        self.rgb_img[:, -1, :] = np.clip(colored_col, 0.0, 1.0)
        
        self.im_sai.set_data(self.rgb_img)
        return [self.im_sai]
    
    def show(self):
        self.ani = animation.FuncAnimation(
            self.fig, self.update_animation, interval=50, blit=False, cache_frame_data=False
        )
        plt.show()

if __name__ == '__main__':
    print("\n" + "="*60)
    print("MANDARIN QUIZ (CSV + AUTO LAUNCH + RANDOM 6)")
    print("="*60)
    
    app = ToneIntroductionQuizMixed()
    app.show()