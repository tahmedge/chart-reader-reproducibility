import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import EC400KDataset
from chart_detection import ChartComponentDetector
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Custom collate for variable annotations
def custom_collate(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, targets

# T5 Chart Reader clearly defined
class ChartReaderT5(nn.Module):
    def __init__(self, model_name='t5-base', hidden_dim=768):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        self.loc_embedding = nn.Linear(4, hidden_dim)
        self.type_embedding = nn.Embedding(8, hidden_dim)
        self.appearance_embedding = nn.Linear(100, hidden_dim)

    def forward(self, text_input, bbox, comp_type, appearance_feat, labels=None):
        tokenized = self.tokenizer(text_input, return_tensors="pt", padding=True).to(bbox.device)

        text_embeds = self.model.encoder.embed_tokens(tokenized.input_ids)
        loc_embeds = self.loc_embedding(bbox)
        type_embeds = self.type_embedding(comp_type)
        appearance_embeds = self.appearance_embedding(appearance_feat)

        combined_embeds = text_embeds + loc_embeds.unsqueeze(1) + type_embeds.unsqueeze(1) + appearance_embeds.unsqueeze(1)

        encoder_outputs = self.model.encoder(inputs_embeds=combined_embeds,
                                             attention_mask=tokenized.attention_mask)

        outputs = self.model(encoder_outputs=encoder_outputs, labels=labels)
        return outputs


# Main training loop clearly integrated
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # EC400K dataset
    dataset_root = "ec400k/cls"
    train_dataset = EC400KDataset(root_dir=dataset_root,
                                  annotation_file="chart_train2019.json",
                                  image_set='train2019')

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              num_workers=4, collate_fn=custom_collate)

    # Load trained detector
    detector = ChartComponentDetector().to(device)
    detector.load_state_dict(torch.load('chart_component_detector_final.pth'))
    detector.eval()

    # Initialize T5
    t5_model = ChartReaderT5().to(device)
    optimizer = torch.optim.Adam(t5_model.parameters(), lr=3e-5)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    num_epochs = 5

    for epoch in range(num_epochs):
        t5_model.train()
        epoch_loss = 0

        for images, targets in train_loader:
            images = images.to(device)

            # Detector inference clearly
            with torch.no_grad():
                types_logits, bbox_preds, heatmaps = detector(images)

            bbox_preds = bbox_preds * 512  # scale bbox
            predicted_types = torch.argmax(types_logits, dim=-1)
            visual_features = heatmaps.mean(dim=[2, 3])

            # Prepare T5 inputs
            text_inputs = ["Summarize the chart."] * images.size(0)
            bbox = bbox_preds.to(device)
            comp_type = predicted_types.to(device)
            appearance_feat = visual_features.to(device)

            # Labels (replace with actual summaries from annotations if available)
            labels_text = ["Placeholder summary."] * images.size(0)
            labels = t5_model.tokenizer(labels_text, return_tensors="pt",
                                        padding=True).input_ids.to(device)

            optimizer.zero_grad()

            outputs = t5_model(text_inputs, bbox, comp_type, appearance_feat, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # Save fine-tuned T5 clearly
    t5_model.model.save_pretrained('t5_chart_reader_finetuned')
    t5_model.tokenizer.save_pretrained('t5_chart_reader_finetuned')
    print("Integrated training completed and T5 model saved.")


if __name__ == '__main__':
    main()

