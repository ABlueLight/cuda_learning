#include <cuda_runtime.h> 
#include <iostream>

//grid has one blob, blob has 1024 threads
// dim3 BlocksperGrid(1);
// dim3 ThreadsperBlock(1024);
__global__ void OneDimAdd(float *d_A, float *d_B, float *d_C, int numElements) {
    int i = threadIdx.x;
    if(i<numElements) {
        d_C[i] = d_A[i] + d_B[i]; 
    }

}


//grid has one blob, blob has (4,256) threads
// dim3 BlocksperGrid(1);
// dim3 ThreadsperBlock(4,256);
__global__ void TwoDimAdd(float *d_A, float *d_B, float *d_C, int numElements) {
    int i = threadIdx.y*blockDim.x + threadIdx.x;
    if(i<numElements) {
        d_C[i] = d_A[i] + d_B[i]; 
    }

}

//grid has (2,3) blob, blob has (2,128) threads
// dim3 BlocksperGrid(2,2);
// dim3 ThreadsperBlock(2,128);
__global__ void TwoandTwoDimAdd(float *d_A, float *d_B, float *d_C, int numElements) {
    int blockindex = gridDim.x*blockIdx.y + blockIdx.x;  
    int i = blockDim.x * blockDim.y* blockindex + threadIdx.y*blockDim.x + threadIdx.x;
    if (i < numElements)
    {
        d_C[i] = d_A[i] + d_B[i];
    }

}
//OneDimAdd、TwoDimAdd、TwoandTwoDimAdd实质上的计算方法都是每个线程处理一个元素的加法
// dim3 BlocksperGrid(1);
// dim3 ThreadsperBlock(256); 每个线程计算四个元素的加法
__global__ void ThreadsDimAdd(float *d_A, float *d_B, float *d_C, int numElements) {
    int i = threadIdx.x;
    if(i<256) {
        for(int j=0;j<4;j++) {
            d_C[i*4+j] = d_A[i*4+j] + d_B[i*4+j];
        }
    }
    

}


int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for(int i=0;i<deviceCount;i++)
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        std::cout << "使用GPU device " << i << ": " << devProp.name << std::endl;
        std::cout << "设备全局内存总量： " << devProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
        std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
        std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "设备上一个线程块（Block）种可用的32位寄存器数量： " << devProp.regsPerBlock << std::endl;
        std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
        std::cout << "设备上多处理器的数量： " << devProp.multiProcessorCount << std::endl;
        std::cout << "======================================================" << std::endl;     
        
    }

    int numElements = 1024;
    float *A = new float[numElements];
    float *B = new float[numElements];
    float *C = new float[numElements];

    for(int i=0;i<numElements;i++) {
        A[i] = i*1.0;
        B[i] = i*1.0;
        C[i] = 0.0;
    }

    int size = numElements*sizeof(float);
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    
    dim3 BlocksperGrid(1);
    dim3 ThreadsperBlock(256);
    
    // int threadsPerBlock = 256;
    // int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    int loop = 10;
//    while(loop-- > 1)
    ThreadsDimAdd<<<BlocksperGrid, ThreadsperBlock>>>(d_A, d_B, d_C, numElements);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    for(int i=0;i<numElements;i++) {
        if(A[i] + B[i] - C[i]>10e-6) {
            std::cout<<"err\n";
        }
    }
    std::cout<<"\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
