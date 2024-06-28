#include "cuda.cuh"
#include <algorithm>
#include <cstring>

// CUDA kernel for calculating the mean and sum of squared deviations
__global__ void calculate_mean_and_deviation(const float* input, size_t N, float* mean, float* sum_of_squares) {
    // Calculate global unique index for each thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) { // Prevent overflow
        atomicAdd(mean, input[idx]);  // Atomically add input value to mean to prevent race conditions
        float deviation = input[idx] - (*mean / (float)N); // Calculate deviation from the mean
        atomicAdd(sum_of_squares, deviation * deviation);  // Atomically add squared deviation to sum of squares
    }
}

// Function to calculate standard deviation using CUDA
float cuda_standarddeviation(const float* const input, const size_t N) {
    float* device_input;
    cudaMalloc((void**)&device_input, N * sizeof(float)); // Allocate memory on GPU
    cudaMemcpy(device_input, input, N * sizeof(float), cudaMemcpyHostToDevice); // Copy input data to GPU

    float mean = 0;
    float* device_mean;
    cudaMalloc((void**)&device_mean, sizeof(float)); // Allocate memory for mean on GPU

    float sum_of_squares = 0;
    float* device_sum_of_squares;
    cudaMalloc((void**)&device_sum_of_squares, sizeof(float)); // Allocate memory for sum of squares on GPU

    // Launch kernel to calculate mean and sum of squares
    calculate_mean_and_deviation <<< (N + 255) / 256, 256 >>> (device_input, N, device_mean, device_sum_of_squares);
    cudaDeviceSynchronize();  // Synchronize to ensure kernel execution is complete
    cudaMemcpy(&mean, device_mean, sizeof(float), cudaMemcpyDeviceToHost); // Copy mean back to host
    cudaMemcpy(&sum_of_squares, device_sum_of_squares, sizeof(float), cudaMemcpyDeviceToHost); // Copy sum of squares back to host

    // Deallocate memory on GPU
    cudaFree(device_input);
    cudaFree(device_mean);
    cudaFree(device_sum_of_squares);

    return sqrtf(sum_of_squares / (float)N); // Calculate and return standard deviation
}

// Clamp function to keep values within a specified range
__device__ unsigned char clamp_cuda_to_range(float value, float min, float max) {
    if (value < min) return (unsigned char)min;
    if (value > max) return (unsigned char)max;
    return (unsigned char)value;
}

// CUDA kernel for image convolution using Sobel filter
__global__ void kernel_image_convolution(const unsigned char* input, unsigned char* output, int width, int height) {
    // Sobel filter kernels for edge detection
    int horizontal_filter[3][3] = {
        { 1, 0, -1},
        { 2, 0, -2},
        { 1, 0, -1}
    };
    int vertical_filter[3][3] = {
        { 1,  2,  1},
        { 0,  0,  0},
        {-1, -2, -1}
    };

    // Calculate global indices for each thread
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;

    if (w >= 1 && w < width - 1 && h >= 1 && h < height - 1) { // Ensure within image boundaries
        float gradient_x = 0;
        float gradient_y = 0;
        // Apply Sobel filters
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int input_offset = (h + i) * width + (w + j);
                gradient_x += input[input_offset] * horizontal_filter[j + 1][i + 1];
                gradient_y += input[input_offset] * vertical_filter[j + 1][i + 1];
            }
        }
        int output_offset = (h - 1) * (width - 2) + (w - 1); // Offset for the output image
        float gradient_magnitude = sqrtf(gradient_x * gradient_x + gradient_y * gradient_y); // Calculate gradient magnitude
        output[output_offset] = clamp_cuda_to_range(gradient_magnitude / 3, 0.0f, 255.0f); // Normalize and clamp the result
    }
}

// Function to perform image convolution using CUDA
void cuda_convolution(const unsigned char* const input, unsigned char* const output, const size_t width, const size_t height) {
    unsigned char* device_image_input;
    unsigned char* device_image_output;

    size_t bytes = width * height * sizeof(unsigned char);
    cudaMalloc(&device_image_input, bytes); // Allocate memory for input image on GPU
    cudaMalloc(&device_image_output, bytes); // Allocate memory for output image on GPU

    cudaMemcpy(device_image_input, input, bytes, cudaMemcpyHostToDevice); // Copy input image to GPU

    dim3 block_size(16, 16); // Define block size
    dim3 num_blocks((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y); // Calculate grid size

    // Launch convolution kernel
    kernel_image_convolution <<< num_blocks, block_size >>> (device_image_input, device_image_output, width, height);

    // Copy result back to host
    cudaMemcpy(output, device_image_output, bytes, cudaMemcpyDeviceToHost);

    // Deallocate memory on GPU
    cudaFree(device_image_input);
    cudaFree(device_image_output);
}

// CUDA kernel for computing histogram
__global__ void compute_histogram(const unsigned int* keys, size_t len_k, unsigned int* histogram) {
    // Calculate global unique index for each thread
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len_k) { // Ensure within bounds
        atomicAdd(&histogram[keys[idx]], 1); // Atomically increment the histogram bin
    }
}

// CUDA kernel for computing boundary indices based on histogram
__global__ void compute_boundaries(const unsigned int* histogram, unsigned int* boundaries, size_t len_b) {
    // Calculate boundary values in serial
    if (threadIdx.x == 0 && blockIdx.x == 0) { // Ensure single thread execution
        for (size_t i = 0; i <= len_b - 2; ++i) {
            boundaries[i + 1] = boundaries[i] + histogram[i]; // Calculate exclusive prefix sum
        }
    }
}

// Function to manage memory and execute histogram and boundary computation using CUDA
void cuda_datastructure(const unsigned int* const keys, const size_t len_k, unsigned int* const boundaries, const size_t len_b) {
    unsigned int* device_keys, * device_boundaries, * device_histogram;
    // Allocate memory on GPU
    cudaMalloc((void**)&device_keys, len_k * sizeof(unsigned int));
    cudaMalloc((void**)&device_boundaries, len_b * sizeof(unsigned int));
    cudaMalloc((void**)&device_histogram, (len_b - 1) * sizeof(unsigned int));

    // Copy input keys to GPU
    cudaMemcpy(device_keys, keys, len_k * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Initialize histogram on GPU
    compute_histogram <<<(len_k + 255) / 256, 256 >>> (device_keys, len_k, device_histogram);

    // Reset boundaries on GPU
    cudaMemset(device_boundaries, 0, len_b * sizeof(unsigned int));

    // Compute boundaries based on histogram
    compute_boundaries <<<1, 1 >>> (device_histogram, device_boundaries, len_b);

    // Copy boundaries back to host
    cudaMemcpy(boundaries, device_boundaries, len_b * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Deallocate memory on GPU
    cudaFree(device_keys);
    cudaFree(device_boundaries);
    cudaFree(device_histogram);
}
