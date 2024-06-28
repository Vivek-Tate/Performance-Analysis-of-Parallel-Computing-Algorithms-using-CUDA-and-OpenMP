#include "openmp.h"
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include <stddef.h>

/**
 * Clamps the given value to the specified range.
 *
 * @param value The value to clamp.
 * @param min The minimum allowable value.
 * @param max The maximum allowable value.
 * @return The clamped value as an unsigned char.
 */
unsigned char clamp_to_range(float value, float min, float max) {
    if (value < min) return (unsigned char)min;
    if (value > max) return (unsigned char)max;
    return (unsigned char)value;
}

float openmp_standarddeviation(const float* const input, const size_t N) {
    float mean = 0.0f;                  // Initialize mean to 0
    float sum_squares = 0.0f;           // Initialize sum of squared differences to 0
    size_t i;                           // Holds value for i for iteration
    // Calculate mean in parallel
#pragma omp parallel for reduction(+:mean)
    for (i = 0; i < N; i++) {
        mean += input[i];               // Add each element to the mean
    }
    mean /= (float)N;                   // Divide the sum by N to get the final mean
    // Calculate sum of squared differences from mean in parallel
#pragma omp parallel for reduction(+:sum_squares)
    for (i = 0; i < N; i++) {
        float diff = input[i] - mean;   // Calculate the difference from the mean
        sum_squares += diff * diff;     // Square the difference and add to sum_squares
    }
    // Calculate and Return the standard deviation using the sum of squared differences
    return sqrtf(sum_squares / N);
}

void openmp_convolution(const unsigned char* const input, unsigned char* const output, const size_t width, const size_t height) {

    // Define the horizontal and vertical Sobel filters for edge detection
    const int horizontalSobelFilter[3][3] = {
        { 1,  0, -1 },
        { 2,  0, -2 },
        { 1,  0, -1 }
    };
    const int verticalSobelFilter[3][3] = {
        { 1,  2,  1 },
        { 0,  0,  0 },
        {-1, -2, -1 }
    };

    size_t x, y;                                            // Declare variables for pixel coordinates

    // Parallelize the loop with private copies of x and y
    // Collapse the nested loops to enhance the work distribution among threads
#pragma omp parallel for private(x, y) collapse(2)
    for (x = 1; x < width - 1; ++x) {                      // Avoid the border pixels
        for (y = 1; y < height - 1; ++y) {
            unsigned int sumX = 0;                         // Sum of products for the horizontal gradient
            unsigned int sumY = 0;                         // Sum of products for the vertical gradient

            // Apply the Sobel filter within the neighborhood of the pixel
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    size_t inputOffset = (y + i) * width + (x + j);
                    sumX += input[inputOffset] * horizontalSobelFilter[j + 1][i + 1];
                    sumY += input[inputOffset] * verticalSobelFilter[j + 1][i + 1];
                }
            }

            // Calculate the output pixel's offset in the output image
            size_t outputOffset = (y - 1) * (width - 2) + (x - 1);

            // Compute the gradient magnitude
            float gradientMagnitude = sqrtf((float)(sumX * sumX + sumY * sumY));

            // Normalize and clamp the gradient magnitude to the range [0, 255]
            output[outputOffset] = clamp_to_range(gradientMagnitude / 3, 0.0f, 255.0f);
        }
    }
}

void openmp_datastructure(const unsigned int* const keys, const size_t len_k, unsigned int* const boundaries, const size_t len_b) {
    // Allocate memory for the histogram, sized just short of len_b
    const size_t histogram_bytes = (len_b - 1) * sizeof(unsigned int);
    unsigned int* histogram = (unsigned int*)malloc(histogram_bytes);

    memset(histogram, 0, histogram_bytes);                  // Initialize histogram memory to zero

    // Parallel calculation of histogram
    // 'guided' scheduling may improve performance by dynamically adjusting the chunk size
#pragma omp parallel for schedule(guided)
    for (int index = 0; index < len_k; ++index) {

        // Use atomic operation to avoid race conditions when updating histogram counts
#pragma omp atomic
        ++histogram[keys[index]];
    }

    memset(boundaries, 0, len_b * sizeof(unsigned int));    // Initialize boundaries array to zero

    // Sequential calculation of boundaries based on histogram counts
    for (int index = 0; index + 1 < len_b; ++index) {

        // Calculate prefix sum to determine boundaries
        boundaries[index + 1] = boundaries[index] + histogram[index];
    }

    free(histogram);                                        // Free the allocated histogram memory
}
