#include <iostream>
#include <random>
#include <vector>
using namespace std;
//This implementation is a binaryReduction algorithm meaning that it will buffer out values to a power of 2.
//NOT FINISHED

#define N 20000//number of input values
#define R 2//reduction factor
#define F (1+((N-1)/R))//final value is ceiling of N/R

//powerTwo will use every thread to sum two values, then use half of those to sum those values, and so on until sizeOut is reached. 
//sizeIn and sizeOut will be power of 2s
__global__ void reduce(double *a,double *z, int sizeOut){
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid > N/2)return;

    extern __shared__ double subTotals[];
    subTotals[threadIdx.x]=(a[tid*2]+a[tid*2+1])/2;//sum every two values using all threads
    __syncthreads();
    int level=2;
    while ((blockDim.x/level) >= sizeOut){//keep halving values until sizeout remains
        if(threadIdx.x % level==0){//use half threads every iteration
            subTotals[threadIdx.x]=(subTotals[threadIdx.x]+subTotals[threadIdx.x+(level/2)])/2;
        }
        __syncthreads();//we have to sync threads every time here :(
        level = level * 2;
    }
    level = level /2;
    if(threadIdx.x % level==0){
        z[tid/level] = subTotals[threadIdx.x];
    }
}

int main(){  
    ////Uncomment to print input values
    // for(int i =0;i< N;i++){//print values to screen
    //     cout << a[i] << " ";
    // }
    // cout << endl;
    cudaEvent_t start,stop;//start timing clock initialization
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int bufferedA = 1, bufferedZ =1;//size of arrays after buffering
    while(N > bufferedA){//get closest power of 2 greater than A
        bufferedA = bufferedA * 2;
    }
    int bufferSizeA = bufferedA - N;//amount that needs to be buffered
    while(F > bufferedZ){//get closest power of 2 greater than Z
        bufferedZ = bufferedZ * 2;
    }

     //determine how to fit problem dimensions to the GPU 
    int blocksPerGrid;
    int threadsPerBlock;
    int outputsPerBlock;
    if(N <= 2048) {//if all values can be calculated on one block just do that
         blocksPerGrid = 1;
         threadsPerBlock = bufferedA/2;
         outputsPerBlock = bufferedZ;
    }
    else{//otherwise we need to handle reduction across multiple blocks
         threadsPerBlock =1024;
         blocksPerGrid = (bufferedA) / 2048;//block should be size so that blocks * threads =size of A
         outputsPerBlock = 1+ (bufferedZ-1) / blocksPerGrid;//ceilieng of outputs
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
    
    
    double *dev_a,*dev_z;//create device side variables
    cudaMalloc((void**)&dev_a,sizeof(double)*bufferedA);
    cudaMalloc((void**)&dev_z,sizeof(double)*bufferedZ);

    cudaMemcpy(dev_a,a,sizeof(double)*bufferedA,cudaMemcpyHostToDevice);
  
    reduce<<<blocksPerGrid,threadsPerBlock,threadsPerBlock*sizeof(double)>>>(dev_a,dev_z,outputsPerBlock);//reduce array a to b 

    //Inside this if statement is a bunch of conditionals to process massive data into very small sizes.
    if(N>2048 && outputsPerBlock > (F*2){//we need to keep processing
        
        while(outputsPerBlock * blocksPerGrid > 2048){//if
            double *in = dev_z;
            reduce<<<blocksPerGrid,threadsPerBlock,threadsPerBlock*sizeof(double)>>>(dev_z,dev_a,outputsPerBlock);//reduce array a to b 
        }
    }else{
    cudaMemcpy(z,dev_z,sizeof(double)*F,cudaMemcpyDeviceToHost);//copy the final amount of results back
    }
    cudaEventRecord(stop);//stop clock
    
    cout << endl;
    //Uncomment to print output array
    cout << "Reduced Array:" <<endl;
    for(int i =0;i< F;i++){//output final reduced values
        if(i % outputsPerBlock == 0) cout << "|";
        cout << z[i] << " ";
    }
    cout << endl << endl << blocksPerGrid << " blocks used to reduce " << N << " by  " << R << " to get " << F << " values"<< endl;
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    cout << "This run took " << milliseconds << " miliseconds to compute." << endl;
    cudaFree(dev_a);//free mem!
    cudaFree(dev_z);
}
