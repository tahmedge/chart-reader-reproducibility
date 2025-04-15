import os
import json
from tqdm import tqdm
from paddleocr import PaddleOCR

ocr_model = PaddleOCR(use_angle_cls=True, lang='en', det_db_box_thresh=0.5)  

def extract_ocr_paddle(image_path):
    result = ocr_model.ocr(image_path, cls=True)
    lines = []
    for line in result[0]:
        text = line[1][0].strip()
        if text:
            lines.append(text)
    return '\n'.join(lines)

def process_chart_images_paddle(input_dir, output_jsonl, max_files=None):
    data = []
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if max_files:
        files = files[:max_files]

    count_file = 0

    for filename in tqdm(files, desc="Extracting OCR with Paddle"):
        img_path = os.path.join(input_dir, filename)
        text = extract_ocr_paddle(img_path)

        if text:
            data.append({
                "input": f"Extracted OCR from chart image: {filename}",
                "output": text
            })

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Saved {len(data)} OCR entries to {output_jsonl}")
