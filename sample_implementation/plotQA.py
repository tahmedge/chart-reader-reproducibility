import json
import os
import pandas as pd

#Change dirs
json_path = "qa_pairs_V1.json"
image_dir = "png"

with open(json_path, 'r', encoding='utf-8') as f:
    qa_data = json.load(f)

rows = []

for item in qa_data["qa_pairs"]:
    image_index = item["image_index"]
    question = item["question_string"]
    answer = item["answer"]
    
    image_path = os.path.join(image_dir, f"{image_index}.png")

    if os.path.exists(image_path):
        rows.append({
            "imagePath": image_path,
            "input": question,
            "output": answer
        })

df = pd.DataFrame(rows)
#Change Name
df.to_csv("plotqa_test_v1.csv", index=False)

