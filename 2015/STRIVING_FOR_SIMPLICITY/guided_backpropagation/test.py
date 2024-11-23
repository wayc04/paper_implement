from guided_backpropagation import GuidedBackpropReLUModel
from utils.image_process import preprocess_image, deprocess_image
from torchvision import models
from PIL import Image
import numpy as np

def test(input_img_path='cat.jpg',
         output_img_path='result.png',
         model=models.resnet50(pretrained=True),
         device='cuda',
         target_category=None):
    """
    :param input_img_path:  Path to the input image
    :param output_img_path: Path to the output image
    :param model: model to use for computing gradients
    :param device: device to use for computing gradients
    :param target_category: target category for computing gradients
    """

    model = model.eval()

    input_img = Image.open(input_img_path).convert('RGB')
    input_img = np.array(input_img)
    input_img = input_img.astype(np.float32) / 255.0
    input_img = preprocess_image(input_img, mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]).to(device)

    gb_model = GuidedBackpropReLUModel(model=model, device=device)

    gb = gb_model(input_img, target_category=target_category)

    gb_np = deprocess_image(gb)

    image_save = Image.fromarray(gb_np)
    image_save.save(output_img_path)


test()