# An example script of testing among Gaussian and Proportional approaches, as well as Iteratively-Learned Power Allocation algorithm after 2 rounds of training (over MNIST data) when noise_variance=0.0 and num_measurement=10

python compressed_sensing.py --model-types vae_map --num-measurements 10 --noise-std 0.0 --seed-no 2055 --measurement-type gaussian
python compressed_sensing.py --model-types vae_map --num-measurements 10 --noise-std 0.0 --seed-no 2055 --measurement-type gaussian_block_general
python compressed_sensing.py --model-types vae_map --num-measurements 10 --noise-std 0.0 --seed-no 2055 --mini-batch 1 --adaptive-round-count 2
