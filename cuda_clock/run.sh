nvcc -o clock64 clock64.cu
srun --account=bcsh-delta-gpu --partition=gpuA100x4-interactive -G 1 -n 1 -N 1 --mem=240G ./clock64