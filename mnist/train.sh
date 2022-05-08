# An example script of Iteratively-Learned Power Allocation algorithm training (over MNIST data) when noise_variance=0.0 and num_measurement=10, with batch-size=3 and num-of-round-2.

#round1
python compressed_sensing.py --model-types vae_map --noise-std 0.0 --num-measurements 10 --seed-no 80 --mini-batch 1 --mini-batch-train 1 --adaptive-round-count 0
python compressed_sensing.py --model-types vae_map --noise-std 0.0 --num-measurements 10 --seed-no 81 --mini-batch 1 --mini-batch-train 1 --adaptive-round-count 0
python compressed_sensing.py --model-types vae_map --noise-std 0.0 --num-measurements 10 --seed-no 82 --mini-batch 1 --mini-batch-train 1 --adaptive-round-count 0 --last-mini-batch 1 --mini-batch-start-seed 80 --mini-batch-end-seed 82

#round2
python compressed_sensing.py --model-types vae_map --noise-std 0.0 --num-measurements 10 --seed-no 83 --mini-batch 1 --mini-batch-train 1 --adaptive-round-count 1
python compressed_sensing.py --model-types vae_map --noise-std 0.0 --num-measurements 10 --seed-no 84 --mini-batch 1 --mini-batch-train 1 --adaptive-round-count 1
python compressed_sensing.py --model-types vae_map --noise-std 0.0 --num-measurements 10 --seed-no 85 --mini-batch 1 --mini-batch-train 1 --adaptive-round-count 1 --last-mini-batch 1 --mini-batch-start-seed 83 --mini-batch-end-seed 85

