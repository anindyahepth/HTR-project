import os
import re
import sys
import argparse
import ast
from collections import OrderedDict

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import utils
#from data import dataset
from model import HTR_VT


def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    transform_fn = transforms.Compose([
        transforms.Resize(tuple([64, 512])),
        transforms.ToTensor()
    ])
    image_tensor = transform_fn(image).unsqueeze(0)
    return image_tensor
 

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nb_cls', type=int, default=90)
    parser.add_argument('--img-size', default=[512, 64], type=int, nargs='+')
    #parser.add_argument('--data_path', type=str, default='/content/HTR-VT/data/read2016/lines/')
    parser.add_argument('--pth_path', type=str, default='/content/HTR-VT/best_CER.pth')
    #parser.add_argument('--train_data_list', type=str, default='/content/HTR-VT/data/read2016/train.ln')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--dict_path', type=str, default='/content/HTR-VT/dict_alph')
    parser.add_argument('--image_path', type=str, default='/content/HTR-VT/data/dida/10000/10000/1/1_10.jpg')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1])
    ckpt = torch.load(args.pth_path, map_location='cpu', weights_only = True)

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

    
    alpha = dict_from_file_to_list(args.dict_path)
    converter = utils.CTCLabelConverter(alpha)

    image_tensor = preprocess_image(args.image_path)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        preds = model(image_tensor)
        preds = preds.float()
        preds_size = torch.IntTensor([preds.size(1)])
        preds = preds.permute(1, 0, 2).log_softmax(2)
        _, preds_index = preds.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        preds_str = converter.decode(preds_index.data, preds_size.data)
        recognized_text = preds_str[0]

  
    #print(preds.max(2))
    print(f"Recognized_text: {recognized_text}")


if __name__ == '__main__':
    main()
