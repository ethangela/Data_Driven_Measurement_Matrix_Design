# Data-Driven Algorithms for Gaussian Measurement Matrix Design in Compressive Sensing

This repository provides code to reproduce results of the paper: [Data-Driven Algorithms for Gaussian Measurement Matrix Design in Compressive Sensing](https://ieeexplore.ieee.org/document/9747617).


### Requirements: 
---
1. Python 3.0
2. Tensorflow v1.15.0
3. PyTorch 1.11.0


### Reproducing quantitative results
---
1. We build a pair of training & testing demo for all experiments ove three dataset by runing two following scripts:
     - ```$ ./mnist/train.sh```
     - ```$ ./mnist/test.sh```  
2. For all experiments, noise variance and #measurements can be set with two following parameters:
     - ```--noise-std``` the variance (square of standard deviation) of noise
     - ```--num-measurements ``` the number of measurements 
3. For MNIST experiment, the tunable parametres are:  
     - ```--seed-no``` the index of image being processed 
     - ```--adaptive-round-count``` the index of current round of training
4. For CelebA dataset, the tunable parametres are:  
     - ```--img-no``` the index of image being processed 
     - ```--load-img-no``` the index of image being processed in previous round
5. For Sythentic dataset, the tunable parametres are:  
     - ```--input_dim``` the dimension of to-be-built synthetic data  


### Additional Information
---
For pre-trained model checkpoints download, see [here](https://to_be_update).


