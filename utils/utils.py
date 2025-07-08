
import torch
import torch.distributed as dist
from torch.distributions.uniform import Uniform

import os
import re
import editdistance
import sys
import math
import logging
import ast
from copy import deepcopy
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Function to load the dictionary (text-to-class map) from a text file

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


def randint(low, high):
    return int(torch.randint(low, high, (1, )))


def rand_uniform(low, high):
    return float(Uniform(low, high).sample())


#Learning rate scheduler

def update_lr_cos(nb_iter, warm_up_iter, total_iter, max_lr, optimizer, min_lr=1e-7):

    if nb_iter < warm_up_iter:
        current_lr = max_lr * (nb_iter + 1) / (warm_up_iter + 1)
    else:
        current_lr = min_lr + (max_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * nb_iter / (total_iter - warm_up_iter)))

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

#CTC converter class

class CTCLabelConverter(object):
    def __init__(self, character):
        dict_character = list(character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1
        # if len(self.dict) == 87:     # '[' and ']' are not in the test set but in the training and validation sets.
        #     self.dict['['], self.dict[']'] = 88, 89
        self.character = ['[blank]'] + dict_character
        # self.dict['[blank]'] = 0
        self.output_dict = self.dict

    def get_dict(self):
        return self.output_dict

    def get_char(self):
        return self.character

    def encode(self, text):
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text).to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        texts = []
        index = 0

        for l in length:
            t = text_index[index:index + l]
            char_list = []
            for i in range(l):
                if t[i]!=0 and (not (i > 0 and t[i - 1] == t[i])) and t[i]<len(self.character):
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts



#------Evaluation metrics--------

def clean_text(text):
  text = re.sub(r'[^\w\s\']', '', text)  # Remove punctuation except apostrophes
  text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
  
  return text.strip()

def format_string_for_wer(str):
    str = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', str)
    str = re.sub('([ \n])+', " ", str).strip()
    return str

import editdistance

def compute_dist_length(pred_ids, target_str, converter):
    pred_str= converter.decode(pred_ids.tolist(), length = [pred_ids.shape[0]])

    pred_str = ''.join(pred_str)
    target_str = ''.join(target_str)

    cer_distance = editdistance.eval(pred_str, target_str)
    target_length = len(target_str)

    pred_words = format_string_for_wer(pred_str).split()
    target_words = format_string_for_wer(target_str).split()

    wer_distance = editdistance.eval(pred_words, target_words)
    target_word_len = len(target_words)

    return int(cer_distance), int(target_length), int(wer_distance) , int(target_word_len)




def calc_metric_loader(data_loader, model, device, converter):

    model.eval()
    model.to(device)


    cer_ed =0
    cer_len_tot = 0
    wer_ed =0
    wer_len_tot = 0

    for i, batch in enumerate(data_loader):
        image = batch['pixel_values'].to(device)
        labels = batch['labels']

        output = model(image, 0.0, 1 , use_masking=False)
        pred_ids = torch.argmax(output, dim=-1)

        for j in range(output.shape[0]):
            output_ids = pred_ids[j]
            target_str = labels[j]
            cer_dist, cer_len, wer_dist, wer_len = compute_dist_length(output_ids, target_str, converter)
            cer_ed += cer_dist
            cer_len_tot += cer_len
            wer_ed += wer_dist
            wer_len_tot += wer_len



    cer = cer_ed / float(cer_len_tot) if cer_len_tot > 0 else float('inf')
    wer = wer_ed / float(wer_len_tot) if wer_len_tot > 0 else float('inf')

    return cer, wer

# def compute_metric(pred_ids, target_ids, converter):
#     # length_pred = [pred_ids.shape[1] for j in range(pred_ids.shape[0])]
#     # length_target = [target_ids.shape[1] for j in range(target_ids.shape[0])]
#     target_ids = target_ids[target_ids != -100]
#     pred_str= converter.decode(pred_ids, length = pred_ids.shape)
#     target_str = converter.decode(target_ids, length = target_ids.shape)
#     pred_str = ''.join(pred_str)
#     target_str = ''.join(target_str)

#     edit_distance = editdistance.eval(target_str, pred_str)
#     target_length = len(target_str)
#     if target_length == 0:
#         return 0.0 if len(pred_str) == 0 else float('inf')
#     cer = float(edit_distance) / target_length

#     # pred_words = clean_text(pred_str).split()
#     # target_words = clean_text(target_str).split()
#     pred_words = format_string_for_wer(pred_str).split()
#     target_words = format_string_for_wer(target_str).split()

#     wer_distance = editdistance.eval(target_words, pred_words)
#     wer = float(wer_distance) / len(target_words) if target_words else 0.0

#     return {"cer": cer, "wer": wer, "pred_str": pred_str, "target_str": target_str, "pred_words":pred_words}


# def calc_metric_loader(data_loader, model, device, converter):
#     model.eval()
#     cer =[]
#     wer =[]

#     for i, batch in enumerate(data_loader):
#         image = batch['pixel_values'].to(device)
#         model.to(device)
#         output = model(image, 0.0, 1 , use_masking=False)
#         pred_ids = torch.argmax(output, dim=-1)

#         for j in range(output.shape[0]):
#             output_ids = pred_ids[j]
#             input_ids = batch['labels'][j]
#             metric = compute_metric(output_ids, input_ids, converter)
#             cer.append(metric['cer'])
#             wer.append(metric['wer'])

#     num_iters = len(cer)
#     print(num_iters)
#     sum_cer = sum(cer)
#     sum_wer = sum(wer)

#     return sum_cer / num_iters, sum_wer / num_iters

#-------CTC Loss Function --------------

def compute_loss(model, input_dict, batch_size, criterion, device):

    pixel_values = input_dict['pixel_values'].to(device)
    labels = input_dict['labels']

    preds = model(pixel_values, 0.4, 8, use_masking=False) # Pass labels to get decoder outputs

    preds = preds.permute(1, 0, 2).log_softmax(2)

    preds_size = torch.IntTensor([preds.size(0)] * batch_size).to(device) # preds.size(0) is now sequence_length

    text_for_loss, length_for_loss = converter.encode(labels)
    text_for_loss, length_for_loss = text_for_loss.to(device), length_for_loss.to(device)

    loss = criterion(preds, text_for_loss, preds_size, length_for_loss).mean()

    return loss
    

def compute_loss_loader(model, data_loader, criterion, device):

  loss_total = 0
  batch_size = data_loader.batch_size

  for i, input in enumerate(data_loader):
        pixel_values = input['pixel_values'].to(device)
        labels = input['labels']
        input_dict = {'pixel_values': pixel_values, 'labels': labels}
        loss = compute_loss(model, input_dict, batch_size, criterion, device)
        loss_total += loss.item()

  return loss_total / len(data_loader)




# def compute_loss(model, input_dict, batch_size, criterion, device):

#     pixel_values = input_dict['pixel_values'].to(device)
#     labels = input_dict['labels'].to(device)

#     # For CTC, we need (sequence_length, batch_size, vocab_size)
#     preds = model(pixel_values, 0.4, 8, use_masking=False) 

#     preds = preds.permute(1, 0, 2).log_softmax(2)

#     preds_size = torch.IntTensor([preds.size(0)] * batch_size).to(device) # preds.size(0) is now sequence_length

#     target_sequences_flat = []
#     target_lengths = []

#     # Iterate through each item in the batch
#     for i in range(batch_size): # Iterate over batch dimension
#         current_labels = labels[i]
#         non_padded_labels = [
#             label_id for label_id in current_labels.tolist()
#             if label_id != -100
#         ]

#         target_sequences_flat.extend(non_padded_labels)
#         target_lengths.append(len(non_padded_labels))

#     targets_for_ctc = torch.IntTensor(target_sequences_flat).to(device)
    
#     lengths_for_ctc = torch.IntTensor(target_lengths).to(device)

    
#     loss = criterion(preds, targets_for_ctc, preds_size, lengths_for_ctc).mean()

#     return loss
    

# def compute_loss_loader(model, data_loader, criterion, device):

#   loss_total = 0
#   batch_size = data_loader.batch_size

#   for i, input in enumerate(data_loader):
#         pixel_values = input['pixel_values'].to(device)
#         labels = input['labels'].to(device)
#         input_dict = {'pixel_values': pixel_values, 'labels': labels}
#         loss = compute_loss(model, input_dict, batch_size, criterion, device)
#         loss_total += loss.item()

#   return loss_total / len(data_loader)



def rgb_to_grayscale(rgb_tensor):

  if rgb_tensor.shape[-3] != 3:
    raise ValueError("Input tensor must have 3 channels (RGB) in the channel dimension.")

  # Grayscale conversion weights
  # These weights are standard for converting RGB to luminance (grayscale)
  # R: 0.2989, G: 0.5870, B: 0.1140
  # Ensure the weights are on the same device as the input tensor
  weights = torch.tensor([0.2989, 0.5870, 0.1140], device=rgb_tensor.device, dtype=rgb_tensor.dtype)

  # Reshape weights for broadcasting: (3,) -> (3, 1, 1) for (C, H, W) input
  # or (1, 3, 1, 1) for (B, C, H, W) input
  if rgb_tensor.dim() == 3: # (C, H, W)
    grayscale_tensor = torch.sum(rgb_tensor * weights.view(3, 1, 1), dim=0, keepdim=True)
  elif rgb_tensor.dim() == 4: # (B, C, H, W)
    grayscale_tensor = torch.sum(rgb_tensor * weights.view(1, 3, 1, 1), dim=1, keepdim=True)
  else:
    raise ValueError("Input tensor must be 3D (C, H, W) or 4D (B, C, H, W).")

  return grayscale_tensor
