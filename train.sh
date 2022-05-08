### MNIST ###

# An example script of Iteratively-Learned Power Allocation algorithm training when noise_variance=0.0 and num_measurement=10, with batch-size=3 and num-of-round=2:
#generate variance maps
python./mnist/variance_map_produce.map
#round 1
python ./mnist/main.py --model-types vae_map --noise-std 0.0 --num-measurements 10 --seed-no 80 --mini-batch 1 --mini-batch-train 1 --adaptive-round-count 0
python ./mnist/main.py --model-types vae_map --noise-std 0.0 --num-measurements 10 --seed-no 81 --mini-batch 1 --mini-batch-train 1 --adaptive-round-count 0
python ./mnist/main.py --model-types vae_map --noise-std 0.0 --num-measurements 10 --seed-no 82 --mini-batch 1 --mini-batch-train 1 --adaptive-round-count 0 --last-mini-batch 1 --mini-batch-start-seed 80 --mini-batch-end-seed 82
#round 2
python ./mnist/main.py --model-types vae_map --noise-std 0.0 --num-measurements 10 --seed-no 83 --mini-batch 1 --mini-batch-train 1 --adaptive-round-count 1
python ./mnist/main.py --model-types vae_map --noise-std 0.0 --num-measurements 10 --seed-no 84 --mini-batch 1 --mini-batch-train 1 --adaptive-round-count 1
python ./mnist/main.py --model-types vae_map --noise-std 0.0 --num-measurements 10 --seed-no 85 --mini-batch 1 --mini-batch-train 1 --adaptive-round-count 1 --last-mini-batch 1 --mini-batch-start-seed 83 --mini-batch-end-seed 85




### CelebA ###

# An example script of Iteratively-Learned Power Allocation algorithm training when noise_variance=16.0 and num_measurement=900, with batch-size=1 and num-of-round=5:
#generate variance maps
python./mnist/variance_map_produce.map
#round 1 to 5
python ./celeba/main.py --measurement-type gaussian_data_driven --model-type map --img-no 7 --noise-std 16.0 --num-measurements 900 --adaptive-round-count 0
python ./celeba/main.py --measurement-type gaussian_data_driven --model-type map --img-no 8 --noise-std 16.0 --num-measurements 900 --adaptive-round-count 1 --load-img-no 7
python ./celeba/main.py --measurement-type gaussian_data_driven --model-type map --img-no 9 --noise-std 16.0 --num-measurements 900 --adaptive-round-count 2 --load-img-no 8
python ./celeba/main.py --measurement-type gaussian_data_driven --model-type map --img-no 10 --noise-std 16.0 --num-measurements 900 --adaptive-round-count 3 --load-img-no 9
python ./celeba/main.py --measurement-type gaussian_data_driven --model-type map --img-no 11 --noise-std 16.0 --num-measurements 900 --adaptive-round-count 4 --load-img-no 10


