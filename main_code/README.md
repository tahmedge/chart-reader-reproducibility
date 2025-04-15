**First run the component detector model**

```python chart_component_detector_hourglass.py --mode train --epochs 50 --batch_size 16 --save_dir ./hourglass_checkpoints```

**When the component detector model is trained, run the T5 training code**

```bash
python chart_derendering_comprehension_t5.py \
    --mode train_t5 \
    --train_csv_path chartqa_data/train_augmented.csv \
    --val_csv_path chartqa_data/val_augmented.csv \
    --image_root_dir chartqa_data/images/ \
    --detector_model_path hourglass_checkpoints/chart_detector_epoch_50.pth \
    --output_dir ./t5_checkpoints \
    --t5_model_name google-t5/t5-base \
    --epochs 50 \
    --batch_size 4 \
    --lr 7e-4 \
    --num_workers 4 \
    --save_interval 1
```

**After the T5 training is finished, run the inference code**

```bash
python chart_derendering_comprehension_t5.py \
    --mode infer_t5 \
    --input_csv chartqa_data/test_augmented.csv \
    --load_t5_checkpoint t5_checkpoints/t5_chart_model_best.pth \
    --detector_model_path hourglass_checkpoints/chart_detector_epoch_50.pth \
    --image_root_dir chartqa_data/images/ \
    --t5_model_name t5-base \
    --output_dir t5_output \
    --output_file t5_results.json
```
