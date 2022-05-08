### MNIST ###

# An example script of single data testing (over image #2055) among Gaussian and Proportional approaches, as well as Iterative algorithm after 2 rounds of training, when noise_variance=0.0 and num_measurement=10:
python ./mnist/main.py --model-types vae_map --num-measurements 10 --noise-std 0.0 --seed-no 2055 --measurement-type gaussian
python ./mnist/main.py --model-types vae_map --num-measurements 10 --noise-std 0.0 --seed-no 2055 --measurement-type gaussian_block_general
python ./mnist/main.py --model-types vae_map --num-measurements 10 --noise-std 0.0 --seed-no 2055 --mini-batch 1 --adaptive-round-count 2




### CelebA ###

# An example script of single data testing (over image #6) among Gaussian and Proportional approaches, as well as Iterative algorithm after 5 rounds of training, when noise_variance=16.0 and num_measurement=900:
python ./celeba/main.py --model-type map --img-no 6 --noise-std 16.0 --num-measurements 900 --measurement-type gaussian 
python ./celeba/main.py --model-type map --img-no 6 --noise-std 16.0 --num-measurements 900 --measurement-type gaussian_block_general 
python ./celeba/main.py --model-type map --img-no 6 --noise-std 16.0 --num-measurements 900 --measurement-type gaussian_data_driven --adaptive-round-count 5 --load-img-no 11




### L1AE ####

#An example script of testing (over 2000 data points) among Gaussian and Proportional approaches, as well as Iterative algorithm after 20 rounds of training with batch_size=300, when num_measurement=[20:120:20]:
python ./L1AE/main.py
