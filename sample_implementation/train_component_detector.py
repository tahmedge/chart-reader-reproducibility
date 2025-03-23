import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import EC400KDataset
from chart_detection import ChartComponentDetector

# Custom collate function to handle variable annotations
def custom_collate(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, targets


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # EC400K Dataset and DataLoader
    dataset_root = "ec400k/cls"
    train_dataset = EC400KDataset(root_dir=dataset_root,
                                  annotation_file="chart_train2019.json",
                                  image_set='train2019')

    train_loader = DataLoader(train_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=custom_collate)

    # Model Initialization
    model = ChartComponentDetector().to(device)

    # Loss Functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_bbox = nn.SmoothL1Loss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Training Loop
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, targets in train_loader:
            images = images.to(device)
            optimizer.zero_grad()

            types_logits, bbox_preds, _ = model(images)
            bbox_preds = bbox_preds * 512  # scale clearly to pixel coordinates

            boxes_batch, labels_batch = [], []

            for idx, t in enumerate(targets):
                if t['boxes'].numel() >= 4:
                    boxes_batch.append(t['boxes'][0])
                    labels_batch.append(t['labels'][0])

            if len(boxes_batch) == 0:
                continue

            boxes_batch = torch.stack(boxes_batch).to(device)
            labels_batch = torch.stack(labels_batch).to(device)

            min_len = min(types_logits.size(0), labels_batch.size(0), bbox_preds.size(0), boxes_batch.size(0))

            loss_cls = criterion_cls(types_logits[:min_len], labels_batch[:min_len])
            loss_bbox = criterion_bbox(bbox_preds[:min_len], boxes_batch[:min_len])

            loss = loss_cls + loss_bbox
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # Save trained model checkpoint
    torch.save(model.state_dict(), "chart_component_detector_final.pth")
    print("Training completed and model checkpoint saved.")


if __name__ == '__main__':
    main()

