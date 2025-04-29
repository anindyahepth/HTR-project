import torch
import torch.nn as nn

import os
import re
import json
import valid
import ast 

from utils import utils
from utils import option
from model.ViT_Resnet import MaskedAutoencoderViT
from collections import OrderedDict
from functools import partial
from datasets import load_dataset
from torchvision import transforms
from augmentations_iam import ErosionDilationElasticRandomTransform
from torch.utils.data import Dataset


class HFImageDataset(Dataset):
    
    def __init__(self, dataset, transform=None):
        """
        Args:
            dataset (datasets.Dataset): Hugging Face dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Gets an image and its label from the dataset.

        Args:
            idx (int): Index of the image.

        Returns:
            tuple: (image, label)
        """
        example = self.dataset[idx]
        image = example['image']  # Assuming 'image' column contains PIL Images
        label = example['text']  # Assuming 'label' column

        if self.transform:
            image = self.transform(image)

        return image, label

def create_model_vitmae(nb_cls, img_size, **kwargs):
    model = MaskedAutoencoderViT(nb_cls=96,
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



def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    model = create_model_vitmae(nb_cls=args.nb_cls, img_size=args.img_size[::-1])

    pth_path = args.save_dir + '/best_CER.pth'
    logger.info('loading HWR checkpoint from {}'.format(pth_path))
    
    ckpt = torch.load(pth_path, map_location='cpu', weights_only = True)
    model_dict = OrderedDict()
    if 'model' in ckpt:
        ckpt = ckpt['model']

    unexpected_keys = ['state_dict_ema', 'optimizer']
    for key in unexpected_keys:
        if key in ckpt:
            del ckpt[key]


    model.load_state_dict(ckpt, strict= False)

    # ckpt = torch.load(pth_path, map_location='cpu')
    # model_dict = OrderedDict()
    # pattern = re.compile('module.')

    # for k, v in ckpt['state_dict_ema'].items():
    #     if re.search("module", k):
    #         model_dict[re.sub(pattern, '', k)] = v
    #     else:
    #         model_dict[k] = v

    # model.load_state_dict(model_dict, strict=True)
    
    model = model.to(device)

    logger.info('Loading test loader...')
    # train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)

    # test_dataset = dataset.myLoadDS(args.test_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
    
    dataset_iam = load_dataset("Teklia/IAM-line")

    dataset_iam_test = dataset_iam["test"]
    transform = transforms.Compose([ transforms.Resize((64, 512)),
    #ErosionDilationColorJitterTransform(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    #ErosionDilationElasticRandomTransform(),
    transforms.ToTensor(),
    ])
    test_dataset = HFImageDataset(dataset_iam_test, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.val_bs,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.num_workers)

    #converter = utils.CTCLabelConverter(train_dataset.ralph.values())
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True).to(device)
    alpha = dict_from_file_to_list(args.dict_path)
    converter = utils.CTCLabelConverter(alpha)

    model.eval()
    with torch.no_grad():
        val_loss, val_cer, val_wer, preds, labels = valid.validation(model,
                                                                     criterion,
                                                                     test_loader,
                                                                     converter,
                                                                     device)

    logger.info(
        f'Test. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} ')


if __name__ == '__main__':
    args = option.get_args_parser()
    main()
