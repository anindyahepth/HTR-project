import torch
import torch.nn as nn

import os
import re
import json
import valid
import ast 

from utils import utils
from utils import option
from model.ViT_ResNet import MaskedAutoencoderViT
from collections import OrderedDict
from functools import partial
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset


def create_datasets_HF(data_path = "Teklia/IAM-line"):
  dataset = load_dataset(data_path)

  dataset_train = dataset["train"]
  dataset_val = dataset["validation"]
  dataset_test = dataset["test"]

  return dataset_train, dataset_val, dataset_test


def collate_fn(batch):
    """Collate function to handle padding within a batch."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"pixel_values": pixel_values, "labels": padded_labels}

    
def create_model_vitmae(nb_cls, img_size, **kwargs):
    model = MaskedAutoencoderViT(nb_cls,
                                 img_size=img_size,
                                 embed_dim=768,
                                 depth=4,
                                 num_heads=6,
                                 mlp_ratio=4,
                                 pre_trained=True,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 **kwargs)

    return model


def get_model(pth_path, base_model, device):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = base_model

  ckpt = torch.load(pth_path, map_location='cpu', weights_only = True)
  model.load_state_dict(ckpt['model'])
  model = model.to(device)

  return model

               


def main():

    args = option.get_args_parser()
    torch.manual_seed(args.seed)

    #-----Model and Dataset ---
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = create_model_vitmae(nb_cls=96, img_size=[64, 1024])
    pth_path = args.pth_path
    model = get_model(pth_path, base_model, device)
    model.eval()
    
    
    dataset_iam_train, dataset_iam_val, dataset_iam_test = create_datasets_HF()
    alpha = utils.dict_from_file_to_list(args.dict_path)
    converter = utils.CTCLabelConverter(alpha)

    test_dataset = IAMDataset(dataset_iam_test, converter, augment=True, transform = transforms.ToTensor())
    
    #-----Dataloader------

    test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=100,
    collate_fn=collate_fn,
    num_workers=0,
    drop_last=True,)


    #-----Eval------


    with torch.no_grad():
        test_cer, test_wer = utils.calc_metric_loader(test_loader, model, device, converter)
        print(f"Test CER :", val_cer)
        print(f"Test WER :", val_wer)
    


if __name__ == '__main__':
    args = option.get_args_parser()
    main()
