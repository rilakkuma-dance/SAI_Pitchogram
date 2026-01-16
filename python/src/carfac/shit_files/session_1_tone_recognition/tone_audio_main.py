import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import threading
import sys
from pathlib import Path
import sounddevice as sd
import soundfile as sf

class TonePlayer:
    def __init__(self):
        # Auto-detect audio file paths
        self.tone_files = self._find_tone_files()
        self.is_playing = False
        
        # Create figure - adjusted for 2x2 grid layout
        self.fig = plt.figure(figsize=(10, 7))
        self.fig.patch.set_facecolor('white')
        
        self._setup_interface()
    
    def _find_tone_files(self):
        """Identical to original logic for finding audio files"""
        script_dir = Path(__file__).parent.resolve()
        possible_locations = [
            Path.cwd() / 'carfac-SAI/python/src/carfac/reference',
            script_dir / 'reference',
            script_dir / 'src/carfac/reference',
            script_dir.parent / 'carfac/reference',
        ]
        
        tone_files = {}
        filenames = {1: 'a1_FV1_MP3.wav', 2: 'a2_FV2_MP3.wav', 3: 'a3_FV3_MP3.wav', 4: 'a4_FV1_MP3.wav'}
        
        for location in possible_locations:
            if location.exists():
                temp_files = {}
                all_found = True
                for tone_num, filename in filenames.items():
                    file_path = location / filename
                    if file_path.exists():
                        temp_files[tone_num] = str(file_path)
                    else:
                        all_found = False
                        break
                if all_found:
                    tone_files = temp_files
                    break
        return tone_files if tone_files else {num: name for num, name in filenames.items()}

    def _setup_interface(self):
        # Coordinates for 2x2 grid [left, bottom, width, height]
        # We use these to place the buttons and text
        grid_positions = [
            [0.15, 0.65], # Tone 1 (Top Left)
            [0.55, 0.65], # Tone 2 (Top Right)
            [0.15, 0.35], # Tone 3 (Bottom Left)
            [0.55, 0.35]  # Tone 4 (Bottom Right)
        ]

        self.play_buttons = []

        for i, (x, y) in enumerate(grid_positions):
            tone_num = i + 1
            
            # 1. Add Text Label
            self.fig.text(x + 0.15, y + 0.12, f'Tone {tone_num}', 
                         fontsize=18, ha='center', va='center', weight='bold')
            
            # 2. Add Play Button directly under the label
            play_ax = plt.axes([x, y, 0.30, 0.08])
            play_btn = Button(play_ax, f'Play Tone {tone_num}', color='lightgray', hovercolor='#cccccc')
            play_btn.label.set_fontsize(12)
            play_btn.label.set_weight('bold')
            
            # Connect callback and store reference
            play_btn.on_clicked(self._make_play_callback(tone_num))
            self.play_buttons.append(play_btn)

        # 3. OK button (Centered at bottom, original design preserved)
        ok_ax = plt.axes([0.40, 0.10, 0.20, 0.08])
        self.ok_btn = Button(ok_ax, 'OK', color='lightgray', hovercolor='#cccccc')
        self.ok_btn.label.set_fontsize(16)
        self.ok_btn.label.set_weight('bold')
        self.ok_btn.on_clicked(self.close_window)
        
    def _make_play_callback(self, tone_num):
        return lambda event: self.play_tone(tone_num)
    
    def play_tone(self, tone_num):
        if self.is_playing:
            return
        
        audio_path = self.tone_files.get(tone_num)
        if not audio_path or not Path(audio_path).exists():
            print(f"⚠️ Audio file not found: {audio_path}")
            return
        
        def _play():
            self.is_playing = True
            try:
                audio_data, sample_rate = sf.read(audio_path)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                sd.play(audio_data, sample_rate)
                sd.wait()
            except Exception as e:
                print(f"❌ Error: {e}")
            self.is_playing = False
        
        threading.Thread(target=_play, daemon=True).start()
    
    def close_window(self, event):
        sd.stop()
        plt.close(self.fig)
        self._launch_next_script()
    
    def _launch_next_script(self):
        script_dir = Path(__file__).parent.resolve()
        next_script = script_dir / 'tone_recognition_audio_one_syllable.py'
        if next_script.exists():
            import subprocess
            subprocess.Popen([sys.executable, str(next_script)])

    def show(self):
        plt.show()

if __name__ == '__main__':
    player = TonePlayer()
    player.show()