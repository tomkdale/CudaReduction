# CudaReduction

Learning Module 7 for distributed/parrallel computing class


Simple.cu Creates as many threads as Reduction value and has each thread sum its specific values

powerTwo codes all binary reduction where every thread sums two values, then half of those threads sum two other values, and so on until F values remain

powerTwo1.0.cu works only for 2*threadsPerBlock of 1 block. This means that values over 2048 wont work

powerTwo2.0.cu works for any amount of N and R as long as F is not greater than 1024, the amount of threads in one block.

powerTwo3.0.cu was not finished, but shouldv'e solved the shortcomings of 2.0 by looping the function until all values were reduced as far as wanted.

A few speed comparisons between simple and powerTwo

    
              Simple.cu   PowerTwo2.0     
N=1000&R=2     .027ms   .018ms
N=1000&R=4     .028ms   .019ms
N=1000&R=10    .032ms   .020ms
N=1000&R=1000  .096ms   .031ms
N=100000&R=2   .26ms    .037ms
N=100000&R100   .32ms   .15ms

My PowerTwo binary reduction code works better than the naive approach! I think thats pretty swell