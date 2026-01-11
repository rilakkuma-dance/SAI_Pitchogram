import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.image as mpimg
import sys
from pathlib import Path
import subprocess

class TonePlayer:
    def __init__(self):
        # Auto-detect image file paths
        self.tone_images = self._find_tone_images()
        
        # Create figure
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.patch.set_facecolor('white')
        
        self._setup_interface()
    
    def _find_tone_images(self):
        """Auto-detect tone image files from multiple possible locations"""
        script_dir = Path(__file__).parent.resolve()
        
        # Try multiple possible locations
        possible_locations = [
            script_dir / 'image',
            script_dir / 'images',
            script_dir.parent / 'image',
            script_dir.parent / 'images',
            Path.cwd() / 'image',
            Path.cwd() / 'images',
        ]
        
        tone_images = {}
        filenames = {
            1: 'tone1_SAI.png',
            2: 'tone2_SAI.png',
            3: 'tone3_SAI.png',
            4: 'tone4_SAI.png',
        }
        
        # Try to find images in each location
        for location in possible_locations:
            if location.exists():
                print(f"Checking for images: {location}")
                all_found = True
                temp_images = {}
                
                for tone_num, filename in filenames.items():
                    image_path = location / filename
                    if image_path.exists():
                        temp_images[tone_num] = str(image_path)
                    else:
                        all_found = False
                        break
                
                if all_found:
                    tone_images = temp_images
                    print(f"✓ Found all tone images in: {location}\n")
                    break
        
        if not tone_images:
            print("⚠️ Warning: Could not find tone image files!")
            print("  Will use pattern lines instead")
            print("  To use images, place them in:")
            print(f"  - {script_dir / 'image'}")
            print("\n  Expected filenames:")
            for tone_num, filename in filenames.items():
                print(f"  - {filename}")
            print()
        
        return tone_images
        
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
            
            # Tone box (visual representation area) - Thinner width
            box_ax = self.fig.add_axes([0.10 + i * 0.23, 0.45, 0.05, 0.40])
            box_ax.axis('off')
            
            # Draw tone pattern or display image in the box
            if self.tone_images and (i + 1) in self.tone_images:
                self._display_tone_image(box_ax, i + 1)
            else:
                box_ax.set_xlim(0, 1)
                box_ax.set_ylim(0, 1)
                box_ax.set_facecolor('#f0f0f0')
                self._draw_tone_pattern(box_ax, i + 1, tone_colors[i])
            
            # Tone label below the image
            main_ax.text(x_pos + 0.4, 1.0, f'Tone {i + 1}', 
                        fontsize=16, ha='center', va='center', weight='bold')
        
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
            # Tone 3: Dipping V-shape
            ax.plot([0.2, 0.5, 0.8], [0.4, 0.15, 0.4], color=color, linewidth=6, solid_capstyle='round')
        elif tone_num == 4:
            # Tone 4: Falling
            ax.plot([0.2, 0.8], [0.75, 0.3], color=color, linewidth=6, solid_capstyle='round')
    
    def _display_tone_image(self, ax, tone_num):
        """Display SAI image for the tone"""
        try:
            image_path = self.tone_images[tone_num]
            img = mpimg.imread(image_path)
            
            # Display image without limits
            ax.imshow(img)
            ax.set_aspect('auto')
            
            print(f"✓ Displayed image for Tone {tone_num}: {Path(image_path).name}")
        except Exception as e:
            print(f"Error loading image for Tone {tone_num}: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to pattern
            tone_colors = ['#E53935', '#FFA726', '#7CB342', '#5C6BC0']
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_facecolor('#f0f0f0')
            self._draw_tone_pattern(ax, tone_num, tone_colors[tone_num - 1])
    
    def close_window(self, event):
        """Close the application and launch the next script"""
        print("\n✓ Closing Tone Player")
        plt.close(self.fig)
        
        # Find and launch tone_recognition_SAI_animation_one_syllable.py
        self._launch_next_script()
    
    def _launch_next_script(self):
        """Find and launch tone_recognition_SAI_animation_one_syllable.py"""
        script_dir = Path(__file__).parent.resolve()
        target_filename = 'tone_recognition_SAI_animation_one_syllable.py'
        
        # 1. Check specific absolute path first (as requested)
        specific_path = Path(r"C:\Users\maruk\carfac-SAI\python\src\carfac\session_1_tone_recognition\tone_recognition_SAI_animation_one_syllable.py")

        # Try multiple possible locations
        possible_locations = [
            specific_path,
            script_dir / target_filename,
            script_dir.parent / target_filename,
            script_dir.parent.parent.parent / 'src' / 'carfac' / target_filename,
            Path.cwd() / target_filename,
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
                subprocess.Popen([sys.executable, str(next_script)])
            except Exception as e:
                print(f"❌ Error launching script: {e}")
        else:
            print(f"\n⚠️ Could not find {target_filename}")
            print("  Searched in:")
            for loc in possible_locations:
                print(f"  - {loc}")
    
    def show(self):
        """Display the GUI"""
        plt.show()


if __name__ == '__main__':
    print("="*60)
    print("MANDARIN TONE VISUALIZER")
    print("="*60)
    print("View the tone patterns above.")
    print("Click OK to close the application and proceed.")
    print("="*60 + "\n")
    
    player = TonePlayer()
    player.show()