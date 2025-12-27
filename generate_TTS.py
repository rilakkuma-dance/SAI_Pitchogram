import asyncio
import os
import edge_tts

# Create output directory
OUTPUT_DIR = "mandarin_audio"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Data from your table
# Format: (ID, Word, Pinyin_Readable, Tone_Numbers)
data = [
    (1,  "书",   "shu",      "1"),
    (2,  "女人", "nuren",    "32"),
    (3,  "雄",   "xiong",    "2"),
    (4,  "去",   "qu",       "4"),
    (6,  "喜欢", "xihuan",   "31"),
    (7,  "街道", "jiedao",   "14"),
    (8,  "熊猫", "xiongmao", "21"),
    (9,  "书店", "shudian",  "14"),
    (10, "去年", "qunian",   "42"),
    (11, "中午", "zhongwu",  "13"),
    (12, "老师", "laoshi",   "31"),
    (13, "学校", "xuexiao",  "24"),
    (14, "医院", "yiyuan",   "14"),
    (15, "游戏", "youxi",    "24"),
    (16, "她",   "ta",       "1")
]

# Define Voices (Standard Mandarin Neural Voices)
# Yunxi = Male, Xiaoxiao = Female
VOICE_MALE = "zh-CN-YunxiNeural"
VOICE_FEMALE = "zh-CN-XiaoxiaoNeural"

async def generate_audio():
    print(f"Generating audio files in '{OUTPUT_DIR}'...")
    print("-" * 40)

    for index, (id_num, word, pinyin, tone) in enumerate(data):
        # Logic to mix voices: Even IDs = Male, Odd IDs = Female
        if id_num % 2 == 0:
            voice = VOICE_MALE
            gender_label = "Male"
        else:
            voice = VOICE_FEMALE
            gender_label = "Female"

        # Create the filename: 01_Word_Tone.mp3
        filename = f"{id_num:02d}_{word}_{tone}.mp3"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # Generate audio
        communicate = edge_tts.Communicate(word, voice)
        await communicate.save(filepath)

        print(f"Saved: {filename} \t| Voice: {gender_label} ({voice})")

    print("-" * 40)
    print("Done!")

# Run the async function
if __name__ == "__main__":
    asyncio.run(generate_audio())