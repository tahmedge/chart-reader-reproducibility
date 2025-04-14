import os
import pandas as pd

test_titles_path = "Chart2Text/data/test/testTitle.txt"

single_caption_dir = "Chart2Text/dataset/titles"
single_image_dir = "Chart2TextImages/images/statista"

multi_caption_dir = "Chart2Text/dataset/multiColumn/titles"
multi_image_dir = "Chart2TextImages/multiColumn/images/statista"

with open(test_titles_path, 'r', encoding='utf-8') as f:
    test_titles = set(line.strip() for line in f if line.strip())

matched_titles = set()
test_data = []

for fname in sorted(os.listdir(single_caption_dir)):
    if fname.endswith(".txt"):
        file_id = os.path.splitext(fname)[0]
        caption_path = os.path.join(single_caption_dir, fname)

        with open(caption_path, 'r', encoding='utf-8') as f:
            title = f.read().strip()

        if title in test_titles:
            matched_titles.add(title)

            image_path = os.path.join(single_image_dir, f"{file_id}.png")
            if os.path.exists(image_path):
                test_data.append({
                    "imagePath": image_path,
                    "input": "summarize:",
                    "output": title
                })

unmatched_titles = test_titles - matched_titles

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
                test_data.append({
                    "imagePath": image_path,
                    "input": "summarize:",
                    "output": title
                })

still_unmatched_titles = test_titles - matched_titles

df_test = pd.DataFrame(test_data)
print(df_test.shape)
df_test.to_csv("chart2text_test_data.csv", index=False)

print(f"Matched titles: {len(matched_titles)}")
print(f"Unmatched titles after both checks: {len(still_unmatched_titles)}")

