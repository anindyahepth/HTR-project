# HTR-project

This project explores different transformer-based models for the task of Handwritten Text Recognition (HTR) at the line level. This is a challenging task for transformers which require pre-training on a very large dataset, since clean, annotated data for HTR is very limited. 

We consider different versions of the following basic model for this task - a ViT coupled with a ResNet feature extractor which generates the input tokens and a CTC loss function. The models are trained on the  IAM dataset of handwritten text at the line level involving English alphabets, numbers, punctuations, and special characters — a total of 95 classes of characters with the number of training and validation images being 6482 and 976 respectively. In spite of the small dataset, these models achieve competetive performance. 
 
The following figure compares the performance of a pair of such models with the standard ViT on the metrics CER (Character Error Rate) and WER (Word Error Rate) in course of the training. 


![HRT_cer_wer](https://github.com/user-attachments/assets/1f37f293-3562-4663-8e14-ff02f2436c40)

The model ViT_Resnet_mask implements masking of the input tokens (in a fashion similar to https://arxiv.org/abs/2409.08573), 
while ViT_Resnet does not. The former model performs better on the WER metric in the long-run. 

With a batch_size = 30, and training over 3000 iterations only, the model ViT_Resnet_mask yields the following scores on the validation dataset:

val_CER = 5.2 , val_WER = 16.4. 
