import torch
import clip
import cv2
from glob import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.transform = transform
        self.image_files = sorted(glob(directory + '/image_*.png'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name)
        depth_image = np.array(image)
        depth_75th_percentile = np.percentile(depth_image, 80)
        depth_image[depth_image > depth_75th_percentile] = 0
        image = Image.fromarray(depth_image)
        if self.transform:
            image = self.transform(image)
        return image

# Directory containing the images
directory = "data/hands17/test/images"

# Create a dataset and dataloader
dataset = ImageDataset(directory, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Prepare the text
text = clip.tokenize(["a depth image having a hand", "a depth image without a hand"]).to(device)

hand_count = 0
total_count = 0

# Classify all images in the directory
for images in dataloader:
    images = images.to(device)
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)

        # Compute the similarity between the image and the texts
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = torch.topk(similarity, 1)

    # If the index is 0, then it is a hand
    hand_count += torch.sum(indices == 0).item()
    total_count += images.size(0)

print(f"Total images: {total_count}")
print(f"Images with hand: {hand_count}")
print(f"Rate: {hand_count / total_count}")
