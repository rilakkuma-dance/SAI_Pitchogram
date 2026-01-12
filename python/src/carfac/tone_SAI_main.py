import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.image as mpimg
import sys
from pathlib import Path

class TonePlayer:
    def __init__(self):
        # Auto-detect image file paths
        self.tone_images = self._find_tone_images()
        
        # Create figure - adjusted size for 2x2 grid
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.patch.set_facecolor('white')
        
        self._setup_interface()
    
    def _find_tone_images(self):
        """Logic to find images in local directories"""
        script_dir = Path(__file__).parent.resolve()
        possible_locations = [
            script_dir / 'image',
            script_dir.parent / 'image',
            Path.cwd() / 'image',
            Path.cwd() / 'images',
        ]
        
        tone_images = {}
        filenames = {1: 'tone1_SAI.png', 2: 'tone2_SAI.png', 3: 'tone3_SAI.png', 4: 'tone4_SAI.png'}
        
        for location in possible_locations:
            if location.exists():
                temp_images = {}
                all_found = True
                for tone_num, filename in filenames.items():
                    image_path = location / filename
                    if image_path.exists():
                        temp_images[tone_num] = str(image_path)
                    else:
                        all_found = False
                        break
                if all_found:
                    tone_images = temp_images
                    break
        return tone_images
        
    def _setup_interface(self):
        # Define grid coordinates [left, bottom, width, height]
        # This creates the 2x2 layout seen in your image
        positions = [
            [0.08, 0.62, 0.40, 0.22], # Tone 1 (Top Left)
            [0.55, 0.62, 0.40, 0.22], # Tone 2 (Top Right)
            [0.08, 0.30, 0.40, 0.22], # Tone 3 (Bottom Left)
            [0.55, 0.30, 0.40, 0.22]  # Tone 4 (Bottom Right)
        ]

        for i, pos in enumerate(positions):
            tone_num = i + 1
            
            # Label placement (aligned with the left of the image box)
            self.fig.text(pos[0], pos[1] + pos[3] + 0.02, f'Tone {tone_num}', 
                         fontsize=14, ha='left', va='bottom')

            # Create axis for the spectrogram
            ax = self.fig.add_axes(pos)
            ax.axis('off')
            
            if self.tone_images and tone_num in self.tone_images:
                self._display_tone_image(ax, tone_num)
            else:
                self._draw_fallback_pattern(ax, tone_num)

        # OK button (Original design restored: centered at bottom)
        ok_ax = plt.axes([0.40, 0.10, 0.20, 0.08])
        self.ok_btn = Button(ok_ax, 'OK', color='lightgray', hovercolor='#cccccc')
        self.ok_btn.label.set_fontsize(16)
        self.ok_btn.label.set_weight('bold')
        self.ok_btn.on_clicked(self.close_window)
        
    def _display_tone_image(self, ax, tone_num):
        try:
            img = mpimg.imread(self.tone_images[tone_num])
            ax.imshow(img)
            ax.set_aspect('auto') # Matches the stretching in your reference image
        except Exception as e:
            self._draw_fallback_pattern(ax, tone_num)

    def _draw_fallback_pattern(self, ax, tone_num):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        colors = ['#E53935', '#FFA726', '#7CB342', '#5C6BC0']
        color = colors[tone_num-1]
        if tone_num == 1: ax.plot([0.2, 0.8], [0.75, 0.75], color=color, lw=6)
        elif tone_num == 2: ax.plot([0.2, 0.8], [0.3, 0.75], color=color, lw=6)
        elif tone_num == 3: ax.plot([0.2, 0.5, 0.8], [0.4, 0.15, 0.4], color=color, lw=6)
        elif tone_num == 4: ax.plot([0.2, 0.8], [0.75, 0.3], color=color, lw=6)

    def close_window(self, event):
        plt.close(self.fig)
        self._launch_next_script()

    def _launch_next_script(self):
        script_dir = Path(__file__).parent.resolve()
        next_script = script_dir / 'tone_recognition_SAI_one_syllable.py'
        if next_script.exists():
            import subprocess
            subprocess.Popen([sys.executable, str(next_script)])

    def show(self):
        plt.show()

if __name__ == '__main__':
    player = TonePlayer()
    player.show()