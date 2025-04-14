import os
import pandas as pd

valid_titles_path = "Chart2Text/data/valid/validTitle.txt"

single_caption_dir = "Chart2Text/dataset/titles"
single_image_dir = "Chart2TextImages/images/statista"

multi_caption_dir = "Chart2Text/dataset/multiColumn/titles"
multi_image_dir = "Chart2TextImages/multiColumn/images/statista"

with open(valid_titles_path, 'r', encoding='utf-8') as f:
    valid_titles = set(line.strip() for line in f if line.strip())

matched_titles = set()
valid_data = []

for fname in sorted(os.listdir(single_caption_dir)):
    if fname.endswith(".txt"):
        file_id = os.path.splitext(fname)[0]
        caption_path = os.path.join(single_caption_dir, fname)

        with open(caption_path, 'r', encoding='utf-8') as f:
            title = f.read().strip()

        if title in valid_titles:
            matched_titles.add(title)

            image_path = os.path.join(single_image_dir, f"{file_id}.png")
            if os.path.exists(image_path):
                valid_data.append({
                    "imagePath": image_path,
                    "input": "summarize:",
                    "output": title
                })

unmatched_titles = valid_titles - matched_titles

for fname in sorted(os.listdir(multi_caption_dir)):
    if fname.endswith(".txt"):
        file_id = os.path.splitext(fname)[0]
        caption_path = os.path.join(multi_caption_dir, fname)

        with open(caption_path, 'r', encoding='utf-8') as f:
            title = f.read().strip()

        if title in unmatched_titles:
            matched_titles.add(title)

            image_path = os.path.join(multi_image_dir, f"{file_id}.png")
            if os.path.exists(image_path):
                valid_data.append({
                    "imagePath": image_path,
                    "input": "summarize:",
                    "output": title
                })

still_unmatched_titles = valid_titles - matched_titles

df_valid = pd.DataFrame(valid_data)
print(df_valid.shape)
df_valid.to_csv("chart2text_valid_data.csv", index=False)

print(f"Matched titles: {len(matched_titles)}")
print(f"Unmatched titles after both checks: {len(still_unmatched_titles)}")

