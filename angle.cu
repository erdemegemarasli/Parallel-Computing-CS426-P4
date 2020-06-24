#include<stdio.h>
#include<stdlib.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>


__global__ void calculateDotProductsAndReduceGPU(int *vec1, int *vec2, int *reduced, int numElements)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int numThreads = blockDim.x;
    
    
    sdata[tid] = 0;
    sdata[tid + numThreads] = 0;
    sdata[tid + (numThreads * 2)] = 0;

    

    if (i < numElements)
    {
        //Multiplications for dot products
        
        sdata[tid] = vec1[i] * vec2[i];
        sdata[tid + numThreads] = vec1[i] * vec1[i];
        sdata[tid + (numThreads * 2)] = vec2[i] * vec2[i];
        __syncthreads();

        //Reduction with sequential addressing with reversed loop and threadID-based indexing
        for (int s = blockDim.x/2; s > 0; s>>=1) 
        {
            if (tid < s) 
            {
                sdata[tid] += sdata[tid + s];
                sdata[tid + numThreads] += sdata[tid + numThreads + s];
                sdata[tid + (numThreads * 2)] += sdata[(tid + (numThreads * 2)) + s];
            }
            __syncthreads();
        }
    }

    if (tid == 0)
    {
        atomicAdd(&reduced[0], sdata[tid]);
        atomicAdd(&reduced[1], sdata[tid + numThreads]);
        atomicAdd(&reduced[2], sdata[tid + (numThreads * 2)]);
    }
}

__host__ int* arrayGenerator(int size, int upper, int lower)
{
    int *arr = (int *)calloc(size, sizeof(int));
    for (int i = 0; i < size; i++)
    {
        arr[i] = (rand() % (upper - lower + 1)) + lower;
    }
    return arr;
}

__host__ double radToDegree(double radians) {
    return radians * (180.0 / M_PI);
}


__host__ int calculateDotProductsCPU(int size, int *vec1, int *vec2)
{
    int result = 0;
    for (int i = 0; i < size; i++){
        result += vec1[i] * vec2[i];
    }
    return result;
}

__host__ double calculateAngleCPU(int size, int *vec1, int *vec2)
{
    return radToDegree(acos(calculateDotProductsCPU(size, vec1, vec2) /
     ( (double) sqrt(calculateDotProductsCPU(size, vec1, vec1)) * (double) sqrt(calculateDotProductsCPU(size, vec2, vec2)))));
}

__host__ double convertMs(double end, double start){
    return (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
}

int main(int argc, char **argv) 
{
    int numElements;
    char *fileName;
    //Basically means block size
    int threadsPerBlock;
    clock_t startTime;
    clock_t endTime;
    double timeArrayGeneration = 0;
    double timeForCPU;
    double timeHostToDevice;
    double timeKernel;
    double timeDeviceToHost;

    if (argc == 3)
    {
        numElements = atoi(argv[1]);
        threadsPerBlock = atoi(argv[2]);
        if (numElements < 1)
        {
            printf("N must be positive integer\n");
            return 0;
        }
        if (threadsPerBlock < 1)
        {
            printf("blocksize must be positive integer\n");
            return 0;
        }
    }
    else if (argc == 4)
    {
        numElements = -1;
        threadsPerBlock = atoi(argv[2]);
        fileName = argv[3];
        if (threadsPerBlock < 1)
        {
            printf("blocksize must be positive integer\n");
            return 0;
        }
    }
    else
    {
        printf("Please enter inputs correctly.\n");
        return 0;
    }
    int *vec1;
    int *vec2;
    if (numElements > 0)
    {
        srand(time(0));
        const int upper = 10;
        const int lower = -10;

        startTime = clock();

        vec1 = arrayGenerator(numElements, upper, lower);
        vec2 = arrayGenerator(numElements, upper, lower);

        endTime = clock();
        timeArrayGeneration = convertMs(endTime, startTime);
    }
    else
    {
        FILE *file = fopen(fileName, "r");
        fscanf(file, "%d", &numElements);
        vec1 = (int *)calloc(numElements, sizeof(int));
        vec2 = (int *)calloc(numElements, sizeof(int));
        for (int i = 0; i < numElements; i++)
        {
            fscanf(file, "%d", &vec1[i]);
        }
        for (int i = 0; i < numElements; i++)
        {
            fscanf(file, "%d", &vec2[i]);
        }
    }

    startTime = clock();

    double angleCPU = calculateAngleCPU(numElements, vec1, vec2);
    
    endTime = clock();
    timeForCPU = convertMs(endTime, startTime);

    //Basically means grid size
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    int *reduced = (int *)calloc(3, sizeof(int));

    int *cuda_vec1;
    int *cuda_vec2;
    int *cuda_reduced;

    cudaMalloc((void**)&cuda_vec1, numElements * sizeof(int));

    cudaMalloc((void**)&cuda_vec2, numElements * sizeof(int));

    cudaMalloc((void**)&cuda_reduced, 3 * sizeof(int));

    startTime = clock();

    cudaMemcpy(cuda_vec1,vec1,numElements * sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(cuda_vec2,vec2,numElements * sizeof(int),cudaMemcpyHostToDevice);

    endTime = clock();
    timeHostToDevice = convertMs(endTime, startTime);

    startTime = clock();

    calculateDotProductsAndReduceGPU<<<blocksPerGrid,threadsPerBlock, 3 * threadsPerBlock * sizeof(int)>>>(cuda_vec1, cuda_vec2, cuda_reduced, numElements);

    cudaDeviceSynchronize();

    endTime = clock();
    timeKernel = convertMs(endTime, startTime);

    startTime = clock();

    cudaMemcpy(reduced,cuda_reduced,3 * sizeof(int),cudaMemcpyDeviceToHost);

    endTime = clock();
    timeDeviceToHost = convertMs(endTime, startTime);
    
    double angleGPU = radToDegree(acos(reduced[0] /
    ( (double) sqrt(reduced[1]) * (double) sqrt(reduced[2]))));

    free(vec1);
    free(vec2);
    free(reduced);
    cudaFree(cuda_reduced);
    
    //---------------Print Info
    printf("Info\n");
    printf("-----------------\n");
    printf("Number of elements: %d\n", numElements);
    printf("Number of threads per block: %d\n", threadsPerBlock);
    printf("Number of blocks will be created: %d\n\n", blocksPerGrid);
    printf("Time\n");
    printf("-----------------\n");
    printf("Time for the array generation: %lf ms\n", timeArrayGeneration);
    printf("Time for the CPU function: %lf ms\n", timeForCPU);
    printf("Time for the Host to Device transfer: %lf ms\n", timeHostToDevice);
    printf("Time for the kernel execution: %lf ms\n", timeKernel);
    printf("Time for the Device to Host transfer: %lf ms\n", timeDeviceToHost);
    printf("Total execution time for GPU: %lf ms\n\n", timeKernel + timeHostToDevice + timeDeviceToHost);
    printf("Results\n");
    printf("-----------------\n");
    printf("CPU result: %.3f\n", angleCPU);
    printf("GPU result: %.3f\n", angleGPU);
    
    return 0;
}
