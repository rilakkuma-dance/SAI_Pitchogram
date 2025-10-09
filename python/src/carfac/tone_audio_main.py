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
        
        # Create figure
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.patch.set_facecolor('white')
        
        self._setup_interface()
    
    def _find_tone_files(self):
        """Auto-detect tone audio files from multiple possible locations"""
        script_dir = Path(__file__).parent.resolve()
        
        # Try multiple possible locations
        possible_locations = [
            # Relative to script - reference folder directly
            script_dir / 'reference',
            script_dir / 'tone_perfect',
            script_dir.parent / 'reference',
            script_dir.parent / 'tone_perfect',
            
            # Downloads folder (Windows)
            Path.home() / 'Downloads' / 'tone_perfect_all_mp3' / 'tone_perfect',
            Path.home() / 'Downloads' / 'tone_perfect',
            
            # Desktop (common location)
            Path.home() / 'Desktop' / 'tone_perfect',
            
            # Current directory
            Path.cwd() / 'reference',
            Path.cwd() / 'tone_perfect',
        ]
        
        tone_files = {}
        filenames = {
            1: 'a1_FV1_MP3.mp3',
            2: 'a2_FV2_MP3.mp3',
            3: 'a3_FV3_MP3.mp3',
            4: 'a4_MV1_MP3.mp3',
        }
        
        # Try to find files in each location
        for location in possible_locations:
            if location.exists():
                print(f"Checking: {location}")
                all_found = True
                temp_files = {}
                
                for tone_num, filename in filenames.items():
                    file_path = location / filename
                    if file_path.exists():
                        temp_files[tone_num] = str(file_path)
                    else:
                        all_found = False
                        break
                
                if all_found:
                    tone_files = temp_files
                    print(f"✓ Found all tone files in: {location}\n")
                    break
        
        if not tone_files:
            print("⚠️ Warning: Could not find tone audio files!")
            print("   Please place the files in one of these locations:")
            print(f"   - {script_dir / 'reference'}")
            print(f"   - {script_dir / 'tone_perfect'}")
            print(f"   - {Path.home() / 'Downloads' / 'tone_perfect'}")
            print("\n   Expected filenames:")
            for tone_num, filename in filenames.items():
                print(f"   - {filename}")
            print()
            
            # Return placeholder paths
            tone_files = {tone_num: filename for tone_num, filename in filenames.items()}
        
        return tone_files
        
    def _setup_interface(self):
        # Main axis for layout
        main_ax = self.fig.add_axes([0.05, 0.1, 0.9, 0.8])
        main_ax.set_xlim(0, 4)
        main_ax.set_ylim(0, 3)
        main_ax.axis('off')
        
        # Tone display boxes (top row)
        tone_colors = ['#E53935', '#FFA726', '#7CB342', '#5C6BC0']  # Red, Orange, Green, Blue
        
        for i in range(4):
            x_pos = i
            
            # Tone box (visual representation area)
            box_ax = self.fig.add_axes([0.08 + i * 0.23, 0.55, 0.18, 0.30])
            box_ax.set_xlim(0, 1)
            box_ax.set_ylim(0, 1)
            box_ax.set_facecolor('#f0f0f0')
            box_ax.axis('off')
            
            # Draw tone pattern in the box
            self._draw_tone_pattern(box_ax, i + 1, tone_colors[i])
            
            # Tone label
            main_ax.text(x_pos + 0.5, 1.8, f'Tone {i + 1}', 
                        fontsize=18, ha='center', va='center', weight='bold')
            
            # Play button for each tone
            play_ax = plt.axes([0.08 + i * 0.23, 0.40, 0.18, 0.08])
            play_btn = Button(play_ax, 'Play', color='lightgray', hovercolor='#cccccc')
            play_btn.label.set_fontsize(14)
            play_btn.label.set_weight('bold')
            
            # Store buttons to prevent garbage collection
            if not hasattr(self, 'play_buttons'):
                self.play_buttons = []
            self.play_buttons.append(play_btn)
            
            # Connect button to play function with tone number
            play_btn.on_clicked(self._make_play_callback(i+1))
        
        # OK button (centered at bottom)
        ok_ax = plt.axes([0.40, 0.15, 0.20, 0.08])
        self.ok_btn = Button(ok_ax, 'OK', color='lightgray', hovercolor='#cccccc')
        self.ok_btn.label.set_fontsize(16)
        self.ok_btn.label.set_weight('bold')
        self.ok_btn.on_clicked(self.close_window)
        
    def _draw_tone_pattern(self, ax, tone_num, color):
        """Draw a visual representation of each tone pattern"""
        if tone_num == 1:
            # Tone 1: High flat
            ax.plot([0.2, 0.8], [0.75, 0.75], color=color, linewidth=6, solid_capstyle='round')
        elif tone_num == 2:
            # Tone 2: Rising
            ax.plot([0.2, 0.8], [0.3, 0.75], color=color, linewidth=6, solid_capstyle='round')
        elif tone_num == 3:
            # Tone 3: Dipping (low)
            ax.plot([0.2, 0.5, 0.8], [0.4, 0.20, 0.4], color=color, linewidth=6, solid_capstyle='round')
        elif tone_num == 4:
            # Tone 4: Falling
            ax.plot([0.2, 0.8], [0.75, 0.3], color=color, linewidth=6, solid_capstyle='round')
    
    def _make_play_callback(self, tone_num):
        """Create a callback function for the play button"""
        def callback(event):
            self.play_tone(tone_num)
        return callback
    
    def play_tone(self, tone_num):
        """Play the audio file for the specified tone"""
        if self.is_playing:
            return
        
        audio_path = self.tone_files.get(tone_num)
        
        if not audio_path:
            print(f"⚠️ No audio file configured for Tone {tone_num}")
            return
        
        def _play():
            self.is_playing = True
            print(f"\n🔊 Playing Tone {tone_num}")
            print(f"   File: {Path(audio_path).name}")
            
            try:
                if not Path(audio_path).exists():
                    print(f"⚠️ Audio file not found: {audio_path}")
                    self.is_playing = False
                    return
                
                # Load and play audio
                audio_data, sample_rate = sf.read(audio_path)
                
                # Handle stereo/mono
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)  # Convert stereo to mono
                
                print(f"   Playing... ({len(audio_data) / sample_rate:.2f} seconds)")
                
                sd.play(audio_data, sample_rate)
                sd.wait()
                
                print(f"✓ Playback complete\n")
                
            except Exception as e:
                print(f"❌ Error playing audio: {e}\n")
            
            self.is_playing = False
        
        # Play in separate thread to not block GUI
        threading.Thread(target=_play, daemon=True).start()
    
    def close_window(self, event):
        """Close the application and launch the next script"""
        print("\n✓ Closing Tone Player")
        sd.stop()
        plt.close(self.fig)
        
        # Find and launch understanding_tone_audio.py
        self._launch_next_script()
    
    def _launch_next_script(self):
        """Find and launch understanding_tone_audio.py"""
        script_dir = Path(__file__).parent.resolve()
        
        # Try multiple possible locations
        possible_locations = [
            # Same directory as this script
            script_dir / 'understanding_tone_audio.py',
            
            # Parent directory
            script_dir.parent / 'understanding_tone_audio.py',
            
            # Specific carfac location
            script_dir.parent.parent.parent / 'src' / 'carfac' / 'understanding_tone_audio.py',
            
            # Current working directory
            Path.cwd() / 'understanding_tone_audio.py',
        ]
        
        next_script = None
        for location in possible_locations:
            if location.exists():
                next_script = location
                print(f"\n✓ Found next script: {location}")
                break
        
        if next_script:
            try:
                print(f"🚀 Launching: {next_script.name}\n")
                import subprocess
                subprocess.Popen([sys.executable, str(next_script)])
            except Exception as e:
                print(f"❌ Error launching script: {e}")
        else:
            print("\n⚠️ Could not find understanding_tone_audio.py")
            print("   Searched in:")
            for loc in possible_locations:
                print(f"   - {loc}")
    
    def show(self):
        """Display the GUI"""
        plt.show()


if __name__ == '__main__':
    print("="*60)
    print("MANDARIN TONE PLAYER (Tones 1-4)")
    print("="*60)
    print("Click Play buttons to hear each tone")
    print("Click OK to close the application")
    print("="*60 + "\n")
    
    player = TonePlayer()
    player.show()