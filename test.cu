#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

#define TOLERANCE 1e-3

__global__ void normalizeMatrix(float *matrix, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float colSum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            colSum += matrix[i * size + idx];
        }
        if (colSum != 0.0f)
        {
            for (int i = 0; i < size; i++)
            {
                matrix[i * size + idx] /= colSum;
            }
        }
    }
}

__global__ void expandMatrix(float *matrix, float *tempMatrix, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        for (int i = 0; i < size; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < size; j++)
            {
                sum += matrix[j * size + idx] * matrix[j * size + i];
            }
            tempMatrix[idx * size + i] = sum;
        }
    }
}

__global__ void inflateMatrix(float *matrix, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        for (int j = 0; j < size; j++)
        {
            matrix[idx * size + j] = fminf(matrix[idx * size + j] * matrix[idx * size + j], 1.0e30f);
        }
    }
}

__global__ void calculateDifference(float *matrixA, float *matrixB, int size, float *diff)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0)
    {
        float sumSqDiff = 0.0f;
        for (int i = 0; i < size * size; i++)
        {
            float error = matrixA[i] - matrixB[i];
            sumSqDiff += error * error;
        }
        *diff = sqrtf(sumSqDiff / (size * size));
    }
}

void markovClustering(float *matrix, int size)
{
    float *d_matrix;
    float *d_tempMatrix;
    cudaMalloc((void **)&d_matrix, size * size * sizeof(float));
    cudaMalloc((void **)&d_tempMatrix, size * size * sizeof(float));
    cudaMemcpy(d_matrix, matrix, size * size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_size(256);
    dim3 grid_size((size + block_size.x - 1) / block_size.x);

    // Normalization
    normalizeMatrix<<<grid_size, block_size>>>(d_matrix, size);
    cudaDeviceSynchronize();

    // Initialize variables for convergence check
    float difference = FLT_MAX;
    float *d_difference;
    cudaMalloc((void **)&d_difference, sizeof(float));

    // Expansion-Inflation iterations
    while (difference > TOLERANCE)
    {
        expandMatrix<<<grid_size, block_size>>>(d_matrix, d_tempMatrix, size);
        cudaDeviceSynchronize();

        inflateMatrix<<<grid_size, block_size>>>(d_tempMatrix, size);
        cudaDeviceSynchronize();

        normalizeMatrix<<<grid_size, block_size>>>(d_tempMatrix, size);
        cudaDeviceSynchronize();

        // Calculate difference between matrices
        calculateDifference<<<grid_size, block_size>>>(d_matrix, d_tempMatrix, size, d_difference);
        cudaMemcpy(&difference, d_difference, sizeof(float), cudaMemcpyDeviceToHost);

        // Swap matrices
        float *temp = d_matrix;
        d_matrix = d_tempMatrix;
        d_tempMatrix = temp;
    }

    cudaMemcpy(matrix, d_matrix, size * size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);
    cudaFree(d_tempMatrix);
    cudaFree(d_difference);
}

int main()
{
    int size = 10;
    float matrix[] = {
        0.0f, 1.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.5f, 0.2f, 0.0f, 0.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.8f, 0.0f, 0.3f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.3f, 0.0f, 0.6f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.6f, 0.0f, 0.4f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.4f, 0.0f, 0.9f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.9f, 0.0f, 0.7f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.7f, 0.0f, 0.1f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.0f};

    markovClustering(matrix, size);

    printf("Resulting matrix:\n");
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            printf("%.4f ", matrix[i * size + j]);
        }
        printf("\n");
    }

    return 0;
}
