import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
from datasets import load_dataset
from torchvision import transforms

import os
import json
import valid
from utils import utils
from utils.sam import SAM
from utils import option
from model.ViT_ResNet import MaskedAutoencoderViT
from functools import partial
import argparse
from collections import OrderedDict
import ast
from torch.utils.data import Dataset
from dataset_preprocess import IAMDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import mlflow


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



    

def main():

  with mlflow.start_run(experiment_id=0):


    args = option.get_args_parser()
    torch.manual_seed(args.seed)

    #-----Model and Dataset ---
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model_vitmae(nb_cls=96, img_size=[64, 1024])
    model.to(device)
    
    dataset_iam_train, dataset_iam_val, dataset_iam_test = create_datasets_HF()
    alpha = utils.dict_from_file_to_list(args.dict_path)
    converter = utils.CTCLabelConverter(alpha)

    train_dataset = IAMDataset(dataset_iam_train, converter, augment=True, transform = transforms.ToTensor())
    val_dataset = IAMDataset(dataset_iam_val, converter, augment=True, transform = transforms.ToTensor())
    
    #-----Dataloaders------

    train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=128,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    drop_last=True,)

    val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=8,
    collate_fn=collate_fn,
    num_workers=0,
    drop_last=True,)

    #------Optimizer----------
  
    optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=1e-7, betas=(0.9, 0.99), weight_decay=0.5)
  
    #------Loss function---------------
  
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)

    #---- training loop ----
    
    n_epochs = 60
    global_step = 0
    batch_size = train_loader.batch_size



    train_loss_list = []
    val_loss_list = []
    val_cer_list = []
    val_wer_list = []

    optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=1e-7, betas=(0.9, 0.99), weight_decay=0.5)


    for epoch in range(n_epochs):
      train_loss = 0.0
      examples_seen = 0
      examples_seen_val = 0

      for i, batch in enumerate(train_loader):
        global_step = i + 1 + epoch * len(train_loader)
        optimizer, current_lr = utils.update_lr_cos(global_step, warm_up_iter=1000, total_iter=n_epochs * len(train_loader), max_lr=1e-3, optimizer=optimizer, min_lr=1e-7)

        model.train()
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        input_dict = {'pixel_values': pixel_values, 'labels': labels}
        loss = utils.compute_loss(model, input_dict, batch_size, criterion, device)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        loss = utils.compute_loss(model, input_dict, batch_size, criterion, device)
        loss.backward()
        optimizer.second_step(zero_grad=True)
        optimizer.zero_grad()

        train_loss += loss.item()
        examples_seen += batch["pixel_values"].shape[0]

        if global_step % 50 == 0:
          train_loss_list.append(train_loss/50)
          print(f"Training loss after step {global_step}:", train_loss/50)
          print(f"Learning rate after step {global_step}:", current_lr)
          trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
          print(f"Trainable parameters after step {global_step}:", trainable_params)
          train_loss = 0.0

        if global_step == 1 or global_step % 100 == 0:
          model.eval()
          val_cer = 0.0
          val_wer = 0.0
          with torch.no_grad():
            val_cer, val_wer = utils.calc_metric_loader(val_loader, model, device, converter)
            loss_val = utils.compute_loss_loader(model, val_loader, criterion, device)
            val_cer_list.append(val_cer)
            val_wer_list.append(val_wer)
            val_loss_list.append(loss_val)
            print(f"Validation CER after step {global_step}:", val_cer)
            print(f"Validation WER after step {global_step}:", val_wer)
            print(f"Validation Loss after step {global_step}:", loss_val)
            model.train()



      print(f"Model trained for {epoch + 1} epoch(s)")

      checkpoint_filename_epoch = f'epoch_{epoch}_checkpoint.pth'
      checkpoint_path_epoch = os.path.join('/content/', checkpoint_filename_epoch)
      checkpoint = { 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                    }

      torch.save(checkpoint, checkpoint_path_epoch)
      
      CHECKPOINT_FILENAME = "epoch_model_checkpoint.pth"
      print(f"Logging checkpoint {CHECKPOINT_FILENAME} to MLflow artifacts...")
      mlflow.log_artifact(checkpoint_path_epoch, artifact_path="checkpoints")
      print(f"Checkpoint logged as artifact under 'checkpoints/{CHECKPOINT_FILENAME}'.")


    for i in range(len(val_cer_list)):
        mlflow.log_metrics(
          { "val_cer": val_cer_list[i],
            "val_wer": val_wer_list[i],
            "val_loss": val_loss_list[i]
          },step = i+1
        )

    for j in range(len(train_loss_list)):
        mlflow.log_metrics(
          { "train_loss": train_loss_list[j]
          },step = j+1
        )

    #mlflow.log_param("learning_rate", lr)
    mlflow.log_param("epochs", n_epochs)
    mlflow.log_param("training_batch_size", batch_size)

    # print(f"Logging checkpoint {CHECKPOINT_FILENAME} to MLflow artifacts...")
    # mlflow.log_artifact(FULL_CHECKPOINT_PATH, artifact_path="checkpoints")
    # print(f"Checkpoint logged as artifact under 'checkpoints/{CHECKPOINT_FILENAME}'.")

mlflow.end_run()

if __name__ == '__main__':
    main()
    
