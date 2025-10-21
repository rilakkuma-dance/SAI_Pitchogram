import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
import subprocess
import sys
from pathlib import Path

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Define the tones with their paths
tones = [
    {'name': 'Tone 1: high', 'color': '#E53935', 'path': [(0.15, 0.75), (0.65, 0.75)]},
    {'name': 'Tone 2: rising', 'color': '#FFA726', 'path': [(0.15, 0.25), (0.65, 0.70)]},
    {'name': 'Tone 3: low', 'color': '#7CB342', 'path': [(0.15, 0.25), (0.65, 0.25)]}, 
    {'name': 'Tone 4: falling', 'color': '#5C6BC0', 'path': [(0.15, 0.75), (0.65, 0.30)]},
]

# Draw each tone
for tone in tones:
    path = tone['path']
    x_coords = [p[0] for p in path]
    y_coords = [p[1] for p in path]
    
    # Draw the line
    ax.plot(x_coords, y_coords, color=tone['color'], linewidth=8, solid_capstyle='round')
    
    # Add text label
    ax.text(0.72, y_coords[-1], tone['name'], 
            fontsize=20, fontweight='bold', 
            color=tone['color'], va='center')

# Add title
ax.text(0.5, 0.92, 'Mandarin has 4 tones as below:', 
        fontsize=28, ha='center', transform=ax.transAxes)

# Set axis properties
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Create OK button
ax_button = plt.axes([0.42, 0.05, 0.16, 0.08])
btn = Button(ax_button, 'OK', color='lightgray', hovercolor='gray')

# Define button click event
def on_button_click(event):
    plt.close()  # Close current window
    
    # Find the practice script relative to current location
    script_dir = Path(__file__).parent
    
    # Try multiple possible locations
    possible_scripts = [
        script_dir / 'understanding_tone_practice.py',  # Same directory
        script_dir / 'tone_intro.py',  # Alternative name in same directory
        script_dir.parent / 'understanding_tone_practice.py',  # Parent directory
        script_dir.parent / 'src' / 'understanding_tone_practice.py',  # In src folder
    ]
    
    # Find and launch the first script that exists
    main_script = None
    for script_path in possible_scripts:
        if script_path.exists():
            main_script = script_path
            break
    
    if main_script:
        print(f"✅ Launching: {main_script}")
        subprocess.Popen([sys.executable, str(main_script)])
    else:
        print("⚠️ Could not find practice script!")
        print("Tried these locations:")
        for path in possible_scripts:
            print(f"  - {path}")

btn.on_clicked(on_button_click)

plt.tight_layout()
plt.show()