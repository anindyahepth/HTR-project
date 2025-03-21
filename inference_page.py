import cv2
import numpy as np
import torch
from PIL import Image
import os

from utils import utils
from utils import option
import torch
import torch.nn as nn

from model.HTR_VT import MaskedAutoencoderViT
from collections import OrderedDict
from functools import partial

import ast


def split_handwritten_page(image_path, output_dir="lines", target_size=(512, 64)):
    """
    Splits a handwritten text page into lines using horizontal projection,
    saves each line as an image, and formats them into torch tensors.

    Args:
        image_path (str): Path to the input JPEG image.
        output_dir (str): Directory to save the line images.
        target_size (tuple): Target (width, height) for each line image.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Preprocessing: Binarization
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal Projection
    horizontal_projection = np.sum(binary_img, axis=1)

    # Find line boundaries
    line_starts = []
    line_ends = []
    threshold = np.max(horizontal_projection) / 100  # Adjust threshold as needed
    in_line = False

    for y, projection_value in enumerate(horizontal_projection):
        if projection_value > threshold and not in_line:
            line_starts.append(y)
            in_line = True
        elif projection_value <= threshold and in_line:
            line_ends.append(y)
            in_line = False

    # Handle the case where the last line extends to the bottom
    if in_line:
        line_ends.append(binary_img.shape[0])

    line_images = []

    for i, (start_y, end_y) in enumerate(zip(line_starts, line_ends)):
        line_img = img[start_y:end_y, :]

        # Pad or resize to target size
        pil_img = Image.fromarray(line_img)
        resized_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)

        # Convert to grayscale and normalize
        resized_gray = resized_img.convert("L")
        np_img = np.array(resized_gray) / 255.0

        # Convert to torch tensor
        tensor_img = torch.from_numpy(np_img).float().unsqueeze(0)

        line_images.append(tensor_img)

        # Save line image
        line_filename = os.path.join(output_dir, f"line_{i}.jpg")
        resized_img.save(line_filename)

    return line_images


def create_model_vitmae(nb_cls, img_size, **kwargs):
    model = MaskedAutoencoderViT(nb_cls,
                                 img_size=img_size,
                                 embed_dim=768,
                                 depth=4,
                                 num_heads=6,
                                 mlp_ratio=4,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 **kwargs)
    
    return model


def dict_from_file_to_list(filepath):
    try:
        with open(filepath, 'r') as file:
            dict_str = file.read()
            dictionary = ast.literal_eval(dict_str)
            return dictionary

    except FileNotFoundError:
        print(f"File '{filepath}' not found.")
        return None
    except ValueError as e:
        print(f"Error parsing the file: {e}")
        return None

###########################################################
###########################################################

def make_predictions(image_path, pth_path, dict_path):
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model_vitmae(nb_cls =90, img_size= [64, 512])
    ckpt = torch.load(pth_path, map_location='cpu', weights_only = True)

    model_dict = OrderedDict()
    if 'model' in ckpt:
        ckpt = ckpt['model']

    unexpected_keys = ['state_dict_ema', 'optimizer']
    for key in unexpected_keys:
        if key in ckpt:
            del ckpt[key]


    model.load_state_dict(ckpt, strict= False)
    model = model.to(device)
    model.eval()

    
    alpha = dict_from_file_to_list(dict_path)
    converter = utils.CTCLabelConverter(alpha)

    line_tensors = split_handwritten_page(image_path) #splitting the page into lines
    #line_tensors = line_tensors.to(device)

    preds_list = []

    for i,line_tensor in enumerate(line_tensors):
        image_tensor = line_tensor
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
          preds = model(image_tensor)
          preds = preds.float()
          preds_size = torch.IntTensor([preds.size(1)])
          preds = preds.permute(1, 0, 2).log_softmax(2)
          _, preds_index = preds.max(2)
          preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
          preds_str = converter.decode(preds_index.data, preds_size.data)
          prediction = preds_str[0]
        preds_list.append(prediction)
  
    return preds_list
###########################################################
###########################################################
def main():
    parser = argparse.ArgumentParser()

    
    
    parser.add_argument('--pth_path', type=str, default='/content/HTR-project/best_CER.pth')
    parser.add_argument('--dict_path', type=str, default='/content/HTR-project/dict_alph')
    parser.add_argument('--image_path', type=str, default='/content/HTR-project/Test_page_2.jpg')
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    pred_list = make_predictions(image_path, pth_path, dict_path)

    print(pred_list)

if __name__ == '__main__':
    main()









