#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024
#define THREADS_PER_BLOCK 256

__device__ void timing_square(float x, float *result, unsigned long long *start, unsigned long long *end) {
    *start = clock64(); // Record start time
    *result = x * x;    // Perform computation
    *end = clock64();   // Record end time
}

// Kernel to call the device function
__global__ void compute_square(const float *input, float *output, unsigned long long *timings_start, unsigned long long *timings_end, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        timing_square(input[idx], &output[idx], &timings_start[idx], &timings_end[idx]);
    }
}

int main() {
    // Host memory
    float *h_input, *h_output;
    unsigned long long *h_timings_start, *h_timings_end;

    size_t bytes = N * sizeof(float);
    size_t timings_bytes = N * sizeof(unsigned long long);

    h_input = (float *)malloc(bytes);
    h_output = (float *)malloc(bytes);
    h_timings_start = (unsigned long long *)malloc(timings_bytes);
    h_timings_end = (unsigned long long *)malloc(timings_bytes);

    for (int i = 0; i < N; i++) h_input[i] = i * 1.0f;

    // Device memory
    float *d_input, *d_output;
    unsigned long long *d_timings_start, *d_timings_end;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_timings_start, timings_bytes);
    cudaMalloc(&d_timings_end, timings_bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute_square<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, d_timings_start, d_timings_end, N);

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_timings_start, d_timings_start, timings_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_timings_end, d_timings_end, timings_bytes, cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < 10; i++) {
        printf("h_output[%d] = %f, start = %llu, end = %llu, duration = %llu clocks\n",
                i, h_output[i], h_timings_start[i], h_timings_end[i], h_timings_end[i] - h_timings_start[i]);
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_timings_start);
    cudaFree(d_timings_end);

    free(h_input);
    free(h_output);
    free(h_timings_start);
    free(h_timings_end);

    return 0;
}
