//CUDA reduction algorithm. simple approach
//Tom Dale
//11-20-18


#include <iostream>
#include <random>
using namespace std;
#define N 100//number of input values
#define R 20//reduction factor
#define F (1+((N-1)/R))//how many values will be in the final output


//basicRun will F number of threads go through R number of values and put the average in z[tid]
__global__ void basicRun(double *a,double *z){
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid > F) return;
    double avg=0;
    for(int i= 0;i<R;i++){//get sum of input values in this threads domain
        avg += a[i+tid*R];
    }
    z[tid]=avg/R;//divide sum by total number of input values to get average
}




int main(){ 
    int bufferedSize = N + (N%R);//buffered size is closest evenly divisible by R value that is equal or greater than n
    double *a,*z;
    a = (double*)malloc(sizeof(double)*N);
    z = (double*)malloc(sizeof(double)*F);
    for(int i =0;i< N;i++){//set a to random values
        a[i]= rand() % 10;
        //a[i] = i;
    }

    for(int i = 0;i<(N%R);i++){//wrap around buffer. a will be extended to be evenly split by R.
        a[N+i] =a[i];//added buffer values will be equal to first few variables in the array as stated in problem
    }

    // for(int i =0;i< bufferedSize;i++){//print values to screen
    //     cout << a[i] << " ";
    // }
    // cout << endl;
    
    double *dev_a,*dev_z;//create device side variables
    cudaMalloc((void**)&dev_a,sizeof(double)*bufferedSize);
    cudaMalloc((void**)&dev_z,sizeof(double)*F);

    cudaMemcpy(dev_a,a,sizeof(double)*bufferedSize,cudaMemcpyHostToDevice);


    int gridSize =100;//number of blocks per grid remeber, should be 1 dimension
    int blockSize = 1024 ;//number of threads per block
    basicRun<<<gridSize,blockSize>>>(dev_a,dev_z);

    cudaMemcpy(z,dev_z,sizeof(double)*F,cudaMemcpyDeviceToHost);


    for(int i =0;i< F;i++){//output final reduced values
        cout << z[i] << " ";
    }
    

    cudaFree(dev_a);
    cudaFree(dev_z);


}