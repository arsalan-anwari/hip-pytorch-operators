#include <iostream>
#include <cstdlib>
#include <hip/hip_runtime.h>

__global__ void triu_kernel(float* matrix, int n, int diagonal) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        if (col < row + diagonal) {
            // Set elements below the diagonal to zero
            matrix[row * n + col] = 0.0f;
        }
    }
}

void initialize_matrix(float* matrix, int n) {
    // Initialize the matrix with some values (you can adapt this part)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i * n + j] = static_cast<float>((std::rand() % 10) + 1);
        }
    }
}

void print_matrix(const float* matrix, int n) {
    // Print the matrix (you can adapt this part)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << matrix[i * n + j] << "\t";
        }
        std::cout << std::endl;
    }
}

void triu_hip(float* matrix, int n, int diagonal) {
    float* d_matrix; // Device matrix

    // Allocate device memory
    hipMalloc(&d_matrix, n * n * sizeof(float));

    // Copy data from host to device
    hipMemcpy(d_matrix, matrix, n * n * sizeof(float), hipMemcpyHostToDevice);

    dim3 block(n, n);

    // Launch the kernel
    hipLaunchKernelGGL(triu_kernel, 1, block, 0, 0, d_matrix, n, diagonal);

    // Copy data back from device to host
    hipMemcpy(matrix, d_matrix, n * n * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_matrix);
}

int main() {
    const int n = 3; // Example matrix size
    float* matrix = new float[n * n];

    // Initialize the matrix
    initialize_matrix(matrix, n);

    // Set the diagonal value (e.g., 2 for the third diagonal)
    int diagonal = 1;

    // Call the HIP implementation
    triu_hip(matrix, n, diagonal);

    // Print the modified matrix (upper triangular part)
    std::cout << "Upper Triangular Matrix:" << std::endl;
    print_matrix(matrix, n);

    delete[] matrix;
    return 0;
}
