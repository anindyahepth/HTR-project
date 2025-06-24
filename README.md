# HTR-project

This project explores different transformer-based models for the task of Handwritten Text Recognition (HTR) at the line level. This is a challenging task for transformers which require pre-training on a very large dataset, since clean, annotated data for HTR is very limited. 

We consider different versions of the following basic model for this task - a ViT coupled with a ResNet feature extractor which generates the input tokens, and a CTC loss function. The models are trained on the  IAM dataset of handwritten text at the line level involving English alphabets, numbers, punctuations, and special characters — a total of 95 classes of characters with the number of training and validation images being 6482 and 976 respectively. In spite of the small dataset, these models achieve competetive performance. 

For the most basic model ViT_ResNet, the model ViT_Resnet yields the following test scores after training for 10000 steps (~50 epochs):

test_CER = 7.0 , test_WER = 21.7 . 
 
The following figure compares the performance of a pair of such models with the standard ViT on the metrics CER (Character Error Rate) and WER (Word Error Rate) in course of the training (metrics are computed after every 100 steps of training). 


![HRT_cer_wer](https://github.com/user-attachments/assets/1f37f293-3562-4663-8e14-ff02f2436c40)

The model ViT_Resnet_mask implements masking of the input tokens (in a fashion similar to https://arxiv.org/abs/2409.08573), 
while ViT_Resnet does not. The former model performs better on the WER metric in the long-run. 


