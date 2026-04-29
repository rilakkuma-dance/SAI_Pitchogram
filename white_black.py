import os
from PIL import Image, ImageOps

# フォルダ設定
input_folder = "input_images"
output_folder = "output_bw"

os.makedirs(output_folder, exist_ok=True)

# 画像を一括処理
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 画像を読み込む
        img = Image.open(input_path)

        # グレースケールに変換
        img_gray = img.convert("L")

        # 白背景にするため反転
        img_bw = ImageOps.invert(img_gray)

        # 保存
        img_bw.save(output_path)

print("全画像の変換が完了しました！")