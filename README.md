# Data-Driven Algorithms for Gaussian Measurement Matrix Design in Compressive Sensing

This repository provides code to reproduce results of the paper: [Data-Driven Algorithms for Gaussian Measurement Matrix Design in Compressive Sensing](https://ieeexplore.ieee.org/document/9747617).


### Requirements: 
1. Python 3.0
2. [Tensorflow v1.15.0]
3. [PyTorch 1.11.0]


### Reproducing quantitative results
---
We build a pair of training/testing demo for all experiments ove three dataset by runing two following scripts:
     - ```$ ./mnist/train.sh```
     - ```$ ./mnist/test.sh``` 
   
1. For MNIST dataset, a demo experiment with batch_size=3 and number_of_measurement=2 can be performed with following two scripts: 
     - ```$ ./mnist/train.sh```
     - ```$ ./mnist/test.sh``` 
2. For CelebA dataset, run python ./celeba/main.py
3. For Sythentic dataset, run python ./L1AE/main.py


### Additional Information
---
For pre-trained model checkpoints download, see [here](https://to_be_update).


