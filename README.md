# Deep Learning Face Classification and Verification

This is my code for the Kaggle Competition for the CMU-11785 Introduction to Depp learning. The competition contains two part, which can be seen in [Face Classification](https://www.kaggle.com/c/11-785-s20-hw2p2-classification/) and [Face Verification](https://www.kaggle.com/c/11-785-s20-hw2p2-verification), where I ranked 47/301 and 2/292, respectively. The data for this compeitition can be found in [data](https://www.kaggle.com/c/11-785-s20-hw2p2-classification/data).

The supported loss functions include softmax, asoftmax, amsoftmax, and centerloss. Supported network include resnet28 and modified spherenet.

## Model Architecture

The model used in this competition is modified based on spherenet and center loss is used as the loss function. The backbone is visualized in this [figure](Architecture.png).

## Experiment Settting

During training, the learning rate begins with 0.1 and Cosine Annealing Warm Restarts is used as the scheduler, which restart the learning rate every 20 epochs. The model is trained with batch size of 256 on 1 GPU and the training is finished at 50 epochs. For the center loss, the weight for updating the center is 0.5 and the weight for adding center loss with the cross-entropy loss is 1.

The image size is changed from 32 × 32 to 64 × 64 . During verification, the face is flipped for data augmentation. The feature embeddings for the face and the flipped face are concatenated and the cosine similarity is calculated between the concatenated features of two faces.
