#include <iostream>
#include <random>
using namespace std;
//This implementation is a binaryReduction algorithm meaning that it will buffer out values to a power of 2.
//This version will only work for values 2048 and under so that all values can be fitted on one block. Larger optimizations to follow.


#define N 2047//number of input values
#define R 1024//reduction factor
#define F N/R//how many values will be in the final output

//powerTwo will use every thread to sum two values, then use half of those to sum those values, and so on unitl sizeOut is reached. 
//sizeIn and sizeOut will be power of 2s
__global__ void reduce(double *a,double *z,int sizeIn, int sizeOut){
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid >=sizeIn /2) return;

    __shared__ double subTotals[N/2];
    subTotals[tid]=(a[tid*2]+a[tid*2+1])/2;//sum every two values using all threads
    __syncthreads();
    int level=2;
    while ((sizeIn/level) > sizeOut){//keep halving values until sizeout remains
        if(tid % level==0){//use half threads every iteration
            subTotals[tid]=(subTotals[tid]+subTotals[tid+(level/2)])/2;
        }
        __syncthreads();//we have to sync threads every time here :(
        level = level * 2;
    }
    level = level /2;
    if(tid % level==0){
        z[tid/level] = subTotals[tid];
    }
}

int main(){ 
    int bufferedA = 1, bufferedZ =1;//size of arrays after buffering
    while(N > bufferedA){//get closest power of 2 greater than A
        bufferedA = bufferedA * 2;
    }
    int bufferSizeA = bufferedA - N;//amount that needs to be buffered
    while(F > bufferedZ){//get closest power of 2 greater than Z
        bufferedZ = bufferedZ * 2;
    }

    double *a,*z;//make a and z vectors
    a = (double*)malloc(sizeof(double)*bufferedA);
    z = (double*)malloc(sizeof(double)*bufferedZ);

    for(int i =0;i< N;i++){//Initialize Values
       // a[i]= rand() % 10;
       a[i] = i;
    }

    for(int i = 0;i<bufferSizeA;i++){//wrap around buffer. 
        a[N+i] =a[i];//added buffer values will be equal to first few variables in the array as stated in problem
    }

    for(int i =0;i< bufferedA;i++){//print values to screen
        cout << a[i] << " ";
    }
    cout << endl;

    
    double *dev_a,*dev_z;//create device side variables
    cudaMalloc((void**)&dev_a,sizeof(double)*bufferedA);
    cudaMalloc((void**)&dev_z,sizeof(double)*bufferedZ);

    cudaMemcpy(dev_a,a,sizeof(double)*bufferedA,cudaMemcpyHostToDevice);


    dim3 gridSize(1);//number of blocks per grid remeber, should be 1 dimension
    dim3 blockSize(bufferedA/2);//number of threads per block
    reduce<<<gridSize,blockSize>>>(dev_a,dev_z,bufferedA,bufferedZ);

    cudaMemcpy(z,dev_z,sizeof(double)*bufferedZ,cudaMemcpyDeviceToHost);

    cout << "Reduced Matrix:" <<endl;
    for(int i =0;i< F;i++){//output final reduced values
        cout << z[i] << " ";
    }
    cout << endl;
    cudaFree(dev_a);
    cudaFree(dev_z);

}