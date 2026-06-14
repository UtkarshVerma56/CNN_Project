# data/dataset.py

from torch.utils.data import Dataset
import torchvision.transforms as T


class RVLCDIPDataset(Dataset):

    def __init__(self, hf_dataset, transforms=None):
        self.dataset = hf_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample = self.dataset[idx]

        image = sample["image"].convert("RGB")
        label = sample["label"]

        if self.transforms:
            image = self.transforms(image)

        return image, label


def get_train_transforms(img_size):

    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(5),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])


def get_val_transforms(img_size):

    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])
