# HTR-project

This project explores different transformer-based models for the task of Handwritten Text Recognition (HTR) at the line level. This is a challenging task for these models which require pre-training on a very large dataset, since clean, annotated data for HTR is very limited. 

We consider different versions of the following basic model - a ViT coupled with a ResNet feature extractor which generates the input tokens and a CTC loss function. The models are trained on the  IAM dataset of handwritten text at the line level involving English alphabets, numbers, punctuations, and special characters — a total of 95 classes of characters with the number of training and validation images being 6482 and 976 respectively. In spite of the small dataset, these models achieve competetive performance.  



![HRT_cer_wer](https://github.com/user-attachments/assets/1f37f293-3562-4663-8e14-ff02f2436c40)
