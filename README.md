# Performance Comparison of CPU and GPU Using CUDA and OpenMP

## Overview

This repository contains the implementation of an academic experiment designed to compare the performance of CPU and GPU using CUDA and OpenMP. The experiment involves reimplementing three algorithms in both OpenMP and CUDA to achieve the fastest performance. The three algorithms are:

1. **Standard Deviation Calculation**
2. **Image Convolution using Sobel Operator**
3. **Histogram-Based Data Structure**

This experiment is part of the coursework for the module Parallel Computing with Graphical Processing Units (GPUs). It aims to assess the ability to implement and optimize parallel algorithms using OpenMP and CUDA.

## Requirements

- **CUDA-Enabled GPU**
- **CUDA Toolkit**
- **Visual Studio with CUDA Installed**
- **OpenMP-Compatible Compiler**

## Getting Started

### Cloning the Repository

```sh
git clone https://github.com/Vivek-Tate/GPU-Parallel-Computing-using-CUDA-and-OpenMP.git
```

### Building the Project

Ensure you have CUDA installed and Visual Studio set up with CUDA integration.

1. Open the solution file in Visual Studio.
2. Set the configuration to `Release`.
3. Build the solution.

### Running the Programs

The executables can be run with specific command-line arguments for different algorithms and inputs.

#### Standard Deviation Calculation

```sh
# Using random seed and population size
CPU SD 12 100000 -b

# Using a CSV input file
CPU SD sd_in.csv -b
```

#### Image Convolution

```sh
# Using a PNG input file
CPU C c_in.png

# Optional output file
CPU C c_in.png c_out.png
```

#### Data Structure

```sh
# Using random seed and array length
CPU DS 12 100000

# Using a CSV input file
CPU DS ds_in.csv

# Optional output file
CPU DS ds_in.csv ds_out.csv
```

### CUDA Implementations

For CUDA implementations, ensure you have a CUDA-capable GPU. The command-line arguments follow the same structure as the CPU versions but replace `CPU` with `CUDA`.

```sh
CUDA SD 12 100000 -b
CUDA C c_in.png
CUDA DS ds_in.csv
```

### OpenMP Implementations

Ensure your compiler supports OpenMP. The command-line arguments follow the same structure as the CPU versions but replace `CPU` with `OpenMP`.

```sh
OpenMP SD 12 100000 -b
OpenMP C c_in.png
OpenMP DS ds_in.csv
```

## Experiment Details

### Standard Deviation Calculation

The standard deviation is computed in two main stages using OpenMP and CUDA:
- **Mean Calculation**
- **Sum of Squared Differences Calculation**

### Image Convolution

The convolution algorithm applies the Sobel operator to an image to detect edges. The horizontal and vertical gradients are computed, and the gradient magnitude is calculated.

### Data Structure

A histogram of a sorted integer array is computed, followed by the calculation of boundary indices. This involves atomic operations and guided scheduling in OpenMP and CUDA.

## Notes

- This repository is for an academic experiment and not a complete project.
- The performance of each implementation is benchmarked, and results are documented in the final report in docs.

## License

See `LICENSE` for more information.
