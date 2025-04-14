import os
import pandas as pd
import time

caption_dir = "Chart2Text/dataset/multiColumn/captions"
image_dir = "Chart2TextImages/multiColumn/images/statista"

caption_files = sorted([f for f in os.listdir(caption_dir) if f.endswith(".txt")])

data = []

for filename in caption_files:
    file_id = os.path.splitext(filename)[0]
    image_path = os.path.join(image_dir, f"{file_id}.png")

    if os.path.exists(image_path):
        with open(os.path.join(caption_dir, filename), 'r', encoding='utf-8') as f:
            caption = f.read().strip()

        data.append({
            "imagePath": image_path,
            "input": "summarize:",
            "output": caption
        })

df = pd.DataFrame(data)
print(df.shape)
df.to_csv("chart2text_data_multiColumn.csv", index=False)

