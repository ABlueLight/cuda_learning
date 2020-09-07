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



