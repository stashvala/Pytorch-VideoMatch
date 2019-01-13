import torch
from torchvision import transforms

_transform = transforms.Compose([
             transforms.Resize((256, 456)),
             transforms.ToTensor(),  # normalizes image to 0-1 values
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])


def preprocess(*imgs):
    return torch.stack([_transform(img) for img in imgs]).squeeze()


def l2_normalization(X, dim, eps=1e-12):
    return X / (torch.norm(X, p=2, dim=dim, keepdim=True) + eps)
