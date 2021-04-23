#include <iostream>  
#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <sys/time.h> 
#include <cuda_runtime.h> 
using namespace std;    

#define w 8000

int GetNowMicros() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<int64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}
extern "C" int func(); 
extern "C" int testmatMul();


__global__ void rgb2graycuda(float* input, float* output, int width, int height) { 

    int width_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int height_idx = blockDim.y * blockIdx.y + threadIdx.y;
    if ((height_idx < height) && (width_idx < width)) {
        float r = *(input + height_idx * width * 3 + width_idx * 3);
        float g = *(input + height_idx * width * 3 + width_idx * 3 + 1);
        float b = *(input + height_idx * width * 3 + width_idx * 3 + 2);
        *(output + height_idx * width + width_idx) = 0.299f * r + 0.587f * b + 0.114f * b;
    }


}

void rgb2graycpu(float *input, float *output, int width, int height) {
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            float r = *(input + h * width * 3 + w * 3);
            float g = *(input + h * width * 3 + w * 3 + 1);
            float b = *(input + h * width * 3 + w * 3 + 2);
            *(output + h * width + w) = 0.299f *r + 0.587f *b + 0.114f *b;
        }
    }

}

struct Matrix
{
    int width;
    int height;
    float *elements;
};

void matMul(float * M, float * N, float * P, int width){
    for (int i = 0; i < width; i++){
        for (int j = 0; j < width; j++){
            float sum = 0;
            for (int k = 0; k < width; k++){
                float a = M[i * width + k];
                float b = N[k * width + j];
                sum += a * b;
            }
            P[i * width + j] = sum;
        }
    }
}



int main()    
{    

    int tt1 = GetNowMicros();
    func();    
    int tt2 = GetNowMicros();
    std::cout<<(tt2-tt1)<<"\n";

    tt1 = GetNowMicros();
    testmatMul();
    tt2 = GetNowMicros();
    std::cout<<(tt2-tt1)/1000/1000<<"\n";
    /*
    int width = w;
    int height = w; 
    
    float * m = (float *)malloc (width * height * sizeof (float));
    float * n = (float *)malloc (width * height * sizeof (float));
    float * p = (float *)malloc (width * height * sizeof (float));

    for (int i = 0; i < width * height; i++){
        m[i] = 1.0;
        n[i] = 2.0;
    }

    struct timeval t1,t2;
    printf("begin matMul\n");
    gettimeofday(&t1,NULL);
    double timeuse;

    matMul(m, n, p, w);

    gettimeofday(&t2,NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
    printf("Use Time:%f\n",timeuse);
    */

    return 0;    
}



