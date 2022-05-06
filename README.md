# Data-Driven Algorithms for Gaussian Measurement Matrix Design in Compressive Sensing

This repository provides code to reproduce results of the paper: [Data-Driven Algorithms for Gaussian Measurement Matrix Design in Compressive Sensing](https://ieeexplore.ieee.org/document/9747617).


### Requirements: 
1. Python 3.0
2. [Tensorflow v1.15.0]
3. [PyTorch 1.11.0]


### Reproducing quantitative results
---

1. Create a scripts directory ```$ mkdir scripts```

2. Identfy a dataset you would like to get the quantitative results on. Locate the file ```./quant_scripts/{dataset}_reconstr.sh```.

3. Change ```BASE_SCRIPT``` in ```src/create_scripts.py``` to be the same as given at the top of ```./quant_scripts/{dataset}_reconstr.sh```.

4. Optionally, comment out the parts of ```./quant_scripts/{dataset}_reconstr.sh``` that you don't want to run.

5. Run ```./quant_scripts/{dataset}_reconstr.sh```. This will create a bunch of ```.sh``` files in the ```./scripts/``` directory, each one of them for a different parameter setting.

6. Start running these scripts.
    - You can run ```$ ./utils/run_sequentially.sh``` to run them one by one.
    - Alternatively use ```$ ./utils/run_all_by_number.sh``` to create screens and start proccessing them in parallel. [REQUIRES: gnu screen][WARNING: This may overwhelm the computer]. You can use ```$ ./utils/stop_all_by_number.sh``` to stop the running processes, and clear up the screens started this way.

8. Create a results directory : ```$ mkdir results```. To get the plots, see ```src/metrics.ipynb```. To get matrix of images (as in the paper), run ```$ python src/view_estimated_{dataset}.py```.

9. You can also manually access the results saved by the scripts. These can be found in appropriately named directories in ```estimated/```. Directory name conventions are defined in ```get_checkpoint_dir()``` in ```src/utils.py```


### Miscellaneous
---
For a complete list of images not used while training on celebA, see [here](https://www.cs.utexas.edu/~ashishb/csgm/celebA_unused.txt).


