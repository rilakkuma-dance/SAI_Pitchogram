import asyncio
import os
import edge_tts

# 创建输出目录
OUTPUT_DIR = "mandarin_audio"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 您更新后的数据表
# 格式: (ID, 汉字, 拼音, 声调)
data = [
    # 第一声 (Tone 1)
    (1,  "妈", "ma",   "1"),
    (2,  "天", "tian", "1"),
    (3,  "心", "xin",  "1"),
    (4,  "车", "che",  "1"),
    
    # 第二声 (Tone 2)
    (5,  "麻", "ma",   "2"),
    (6,  "学", "xue",  "2"),
    (7,  "人", "ren",  "2"),
    (8,  "白", "bai",  "2"),
    
    # 第三声 (Tone 3)
    (9,  "马", "ma",   "3"),
    (10, "老", "lao",  "3"),
    (11, "火", "huo",  "3"),
    (12, "狗", "gou",  "3"),
    
    # 第四声 (Tone 4)
    (13, "叫", "jiao", "4"),
    (14, "骂", "ma",   "4"),
    (15, "去", "qu",   "4"),
]

# 定义声音 (标准普通话神经网络语音)
# Yunxi = 男声, Xiaoxiao = 女声
VOICE_MALE = "zh-CN-YunxiNeural"
VOICE_FEMALE = "zh-CN-XiaoxiaoNeural"

async def generate_audio():
    print(f"正在生成音频文件到 '{OUTPUT_DIR}'...")
    print("-" * 40)

    for index, (id_num, word, pinyin, tone) in enumerate(data):
        # 混合声音逻辑: ID为偶数 = 男声, ID为奇数 = 女声
        if id_num % 2 == 0:
            voice = VOICE_MALE
            gender_label = "Male"
        else:
            voice = VOICE_FEMALE
            gender_label = "Female"

        # 创建文件名: 01_汉字_声调.mp3
        # 例如: 01_妈_1.mp3
        filename = f"{id_num:02d}_{word}_{tone}.mp3"
        filepath = os.path.join(OUTPUT_DIR, filename)

        try:
            # 生成音频
            communicate = edge_tts.Communicate(word, voice)
            await communicate.save(filepath)
            
            print(f"已保存: {filename} \t| 声音: {gender_label} ({voice})")
            
            # [关键修正] 暂停 1.5 秒以避免触发 API 速率限制错误
            await asyncio.sleep(1.5)

        except Exception as e:
            print(f"失败: {filename} - 错误信息: {e}")

    print("-" * 40)
    print("全部完成!")

# 运行异步函数
if __name__ == "__main__":
    asyncio.run(generate_audio())