import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from torchvision import transforms

import os
import json
import valid
from utils import utils
from utils import sam
from utils import option
from data import dataset
from model.HTR_VT import MaskedAutoencoderViT
from model.ViT_DW import ViT
from functools import partial
import argparse
from collections import OrderedDict
import ast
from torch.utils.data import Dataset
from augmentations_iam import ErosionDilationElasticRandomTransform
import mlflow

#mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

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
    model = MaskedAutoencoderViT(nb_cls,
                                 img_size=img_size,
                                 embed_dim=768,
                                 depth=4,
                                 num_heads=6,
                                 mlp_ratio=4,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 **kwargs)
    
    return model



def create_model_vitdw(image_size, num_classes):
     model =  ViT(image_size = image_size,
                    patch_size= (64, 32),
                    num_classes = num_classes,
                    dim= 768,
                    depth= 4,
                    heads= 6,
                    mlp_dim= 128 ,
                    dim_head= 64,
                    dropout= 0.0,
                    emb_dropout= 0.0,
                    )
     return model 


def compute_loss(args, model_type, model, image, batch_size, criterion, text, length):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    if model_type == 'vitmae':
       preds = model(image, args.mask_ratio, args.max_span_length, use_masking=True)
        
    elif model_type == 'vitdw':
       preds = model(image)
    
    preds = preds.float()
    # print(f"preds shape: {preds.shape}")
    preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(device)
    preds = preds.permute(1, 0, 2).log_softmax(2)

    torch.backends.cudnn.enabled = False
    loss = criterion(preds, text.to(device), preds_size, length.to(device)).mean()
    torch.backends.cudnn.enabled = True
    return loss

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

  #mlflow.pytorch.autolog()

  with mlflow.start_run(experiment_id=0):

    args = option.get_args_parser()
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    writer = SummaryWriter(args.save_dir)

    model_type = args.model_type

    if model_type == 'vitmae':
       model = create_model_vitmae(nb_cls=args.nb_cls, img_size=args.img_size[::-1])
        
    elif model_type == 'vitdw':
       model = create_model_vitdw(image_size= (64, 512), num_classes=args.nb_cls)


    total_param = sum(p.numel() for p in model.parameters())
    logger.info('total_param is {}'.format(total_param))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model.train()
    model.to(device)
    
    model_ema = utils.ModelEma(model, args.ema_decay)
    model.zero_grad()

    logger.info('Loading train loader...')
    dataset_iam = load_dataset("Teklia/IAM-line")

    dataset_iam_train = dataset_iam["train"]
    transform = transforms.Compose([ transforms.Resize((64, 512)),
    #ErosionDilationColorJitterTransform(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    ErosionDilationElasticRandomTransform(),
    transforms.ToTensor(),
    ])
    train_dataset = HFImageDataset(dataset_iam_train, transform=transform)


    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_bs,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.num_workers,
                                               )
    train_iter = dataset.cycle_data(train_loader)

    logger.info('Loading val loader...')
    dataset_iam_val = dataset_iam["validation"]
    val_dataset = HFImageDataset(dataset_iam_val, transform=transform)
    #val_dataset = dataset.myLoadDS(args.val_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_bs,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.num_workers)

    optimizer = sam.SAM(model.parameters(), torch.optim.AdamW, lr=1e-7, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
    alpha = dict_from_file_to_list(args.dict_path)
    converter = utils.CTCLabelConverter(alpha)

    best_cer, best_wer = 1e+6, 1e+6
    train_loss = 0.0
    val_cer_list, val_wer_list = [], []
    #n_val = int(args.total_iter/args.eval_iter)

    #### ---- train & eval ---- ####

    for nb_iter in range(1, args.total_iter):

        optimizer, current_lr = utils.update_lr_cos(nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer)

        optimizer.zero_grad()
        batch = next(train_iter)
        image = batch[0].to(device)
        text, length = converter.encode(batch[1])
        text, length = text.to(device), length.to(device) #text,length moved to device
        batch_size = image.size(0)
        loss = compute_loss(args, model_type, model, image, batch_size, criterion, text, length)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        compute_loss(args, model_type, model, image, batch_size, criterion, text, length).backward()
        optimizer.second_step(zero_grad=True)
        model.zero_grad()
        model_ema.update(model, num_updates=nb_iter / 2)
        train_loss += loss.item()

        if nb_iter % args.print_iter == 0:
            train_loss_avg = train_loss / args.print_iter

            logger.info(f'Iter : {nb_iter} \t LR : {current_lr:0.5f} \t training loss : {train_loss_avg:0.5f} \t ' )

            writer.add_scalar('./Train/lr', current_lr, nb_iter)
            writer.add_scalar('./Train/train_loss', train_loss_avg, nb_iter)
            train_loss = 0.0

        if nb_iter % args.eval_iter == 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_cer, val_wer, preds, labels = valid.validation(model_ema.ema,
                                                                             criterion,
                                                                             val_loader,
                                                                             converter,
                                                                             device)
                val_cer_list.append(val_cer)
                val_wer_list.append(val_wer)

                if val_cer < best_cer:
                    logger.info(f'CER improved from {best_cer:.4f} to {val_cer:.4f}!!!')
                    best_cer = val_cer
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(args.save_dir, 'best_CER.pth'))

                if val_wer < best_wer:
                    logger.info(f'WER improved from {best_wer:.4f} to {val_wer:.4f}!!!')
                    best_wer = val_wer
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(args.save_dir, 'best_WER.pth'))

                logger.info(
                    f'Val. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} \t ')

                writer.add_scalar('./VAL/CER', val_cer, nb_iter)
                writer.add_scalar('./VAL/WER', val_wer, nb_iter)
                writer.add_scalar('./VAL/bestCER', best_cer, nb_iter)
                writer.add_scalar('./VAL/bestWER', best_wer, nb_iter)
                writer.add_scalar('./VAL/val_loss', val_loss, nb_iter)
                model.train()

    for i in range(len(val_cer_list)):
        	mlflow.log_metrics(
            { "validation_cer": val_cer_list[i],
              "val_wer": val_wer_list[i],
            },step = i+1
        )

mlflow.end_run()

if __name__ == '__main__':
    main()
