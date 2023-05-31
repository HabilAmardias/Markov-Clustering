#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define MAX_ITERATIONS 1

__global__ void normalizeMatrix(float *matrix, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            sum += matrix[i * size + idx];
        }
        for (int i = 0; i < size; i++)
        {
            matrix[i * size + idx] /= sum;
        }
    }
}

__global__ void expandMatrix(float *matrix, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                matrix[idx * size + j] += matrix[idx * size + i] * matrix[i * size + j];
            }
        }
    }
}

__global__ void inflateMatrix(float *matrix, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float column_sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            column_sum += matrix[i * size + idx];
        }
        for (int i = 0; i < size; i++)
        {
            matrix[idx * size + i] = matrix[idx * size + i] * matrix[idx * size + i];
            matrix[idx * size + i] /= column_sum;
        }
    }
}

void markovClustering(float *matrix, int size)
{
    float *d_matrix;
    cudaMalloc((void **)&d_matrix, size * size * sizeof(float));
    cudaMemcpy(d_matrix, matrix, size * size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_size(256);
    dim3 grid_size((size + block_size.x - 1) / block_size.x);

    // Normalization
    normalizeMatrix<<<grid_size, block_size>>>(d_matrix, size);
    cudaDeviceSynchronize();

    // Expansion-Inflation iterations
    for (int i = 0; i < MAX_ITERATIONS; i++)
    {
        expandMatrix<<<grid_size, block_size>>>(d_matrix, size);
        cudaDeviceSynchronize();

        inflateMatrix<<<grid_size, block_size>>>(d_matrix, size);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(matrix, d_matrix, size * size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);
}

int main()
{
    int size = 4;
    float matrix[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f};

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
