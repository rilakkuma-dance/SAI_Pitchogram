import os
from PIL import Image

input_folder = "input_images"
output_folder = "output_bw"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        img = Image.open(input_path)
        bw = img.convert("L")  # convert to grayscale
        bw.save(output_path)

print("Done!")