import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor

def preprocess_image(img: np.ndarray , mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> torch.Tensor:
    """Preprocess image for PyTorch model input."""
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    # 处理图像并增加批量维度
    return preprocessing(img).unsqueeze(0)


def deprocess_image(img: np.ndarray) -> np.ndarray:
    """Deprocess the image array to convert it back to a displayable format."""

    img_tensor = torch.tensor(img).float()


    img_tensor = img_tensor.detach().cpu()
    img_tensor = img_tensor.clone()
    img_tensor = img_tensor - img_tensor.mean()
    img_tensor = img_tensor / (img_tensor.std() + 1e-5)
    img_tensor = img_tensor * 0.1
    img_tensor = img_tensor + 0.5
    img_tensor = torch.clamp(img_tensor, 0, 1)


    return (img_tensor * 255).byte().numpy()


