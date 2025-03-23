import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class EC400KDataset(Dataset):
    def __init__(self, root_dir, annotation_file, image_set='train2019', transforms=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images', image_set)
        self.annotation_path = os.path.join(root_dir, 'annotations', annotation_file)
        self.transforms = transforms if transforms else T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])
        
        with open(self.annotation_path, 'r') as f:
            data = json.load(f)

        self.images_info = data['images']
        self.annotations = data['annotations']

        self.annotations_per_image = {}
        for anno in self.annotations:
            img_id = anno['image_id']
            if img_id not in self.annotations_per_image:
                self.annotations_per_image[img_id] = []
            self.annotations_per_image[img_id].append(anno)

        self.image_id_to_info = {img['id']: img for img in self.images_info}
        self.image_ids = list(self.image_id_to_info.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_id_to_info[image_id]

        image_path = os.path.join(self.image_dir, image_info['file_name'])
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            print(f"Skipping corrupted image: {image_path}")
            # Return an empty dummy sample or next available valid sample
            return self.__getitem__((idx + 1) % len(self))

        image_tensor = self.transforms(image)

        annotations = self.annotations_per_image.get(image_id, [])

        boxes, points, pies, labels, areas = [], [], [], [], []

        for anno in annotations:
            bbox = anno['bbox']
            category_id = anno['category_id']
            area = anno['area']

            if category_id in [1, 4, 5, 6, 7]:
                if len(bbox) == 4:
                    boxes.append(bbox)
                    labels.append(category_id)
                    areas.append(area)
            elif category_id == 2:
                points.append(bbox)
                labels.append(category_id)
                areas.append(area)
            elif category_id == 3:
                if len(bbox) == 6:
                    pies.append(bbox)
                    labels.append(category_id)
                    areas.append(area)

        target = {'image_id': torch.tensor([image_id])}

        target['boxes'] = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        target['points'] = [torch.tensor(pt, dtype=torch.float32) for pt in points] if points else []
        target['pies'] = torch.tensor(pies, dtype=torch.float32) if pies else torch.empty((0, 6), dtype=torch.float32)
        target['labels'] = torch.tensor(labels, dtype=torch.int64)
        target['area'] = torch.tensor(areas, dtype=torch.float32)

        return image_tensor, target


# Example usage
if __name__ == "__main__":
    dataset_root = "ec400k/cls"
    dataset = EC400KDataset(root_dir=dataset_root, annotation_file="chart_train2019.json", image_set='train2019')

    image, target = dataset[0]
    print(f"Image shape: {image.shape}")
    print("Target:", target)

