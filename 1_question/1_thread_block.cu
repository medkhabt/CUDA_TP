
/*

This program will numerically compute the integral of

                  4/(1+x*x) 
				  
from 0 to 1.  The value of this integral is pi -- which 
is great since it gives us an easy way to check the answer.

History: Written by Tim Mattson, 11/1999.
         Modified/extended by Jonathan Rouzaud-Cornabas, 10/2022
*/

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <algorithm>

#define __NUM_STEPS__ 100000  
static long num_steps = 100000000;
double *step = (double*)malloc(sizeof(double));
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
__global__ void atomic_pi(double *A, double *result , long *size) {
    double step = 1.0/(double)(*size);
   A[blockIdx.x] = (( blockIdx.x - 0.5 ) * (step)) * (( blockIdx.x - 0.5 ) * (step));   
   atomicAdd(result, A[blockIdx.x]); 
}
__global__ void atomic_pi_without_step(double *A, double *result) {
    double step = 1.0/(double) __NUM_STEPS__; 
    double x = ( blockIdx.x - 0.5 ) * (step);  
    atomicAdd(result, ((double)4.0 / (double)(1.0 + x * x))) ;   
// TODO remove the comment after
}
__global__ void test(double *A, double *result) {

   A[blockIdx.x]++;  
   atomicAdd(result, 10);
}

int main (int argc, char** argv)
{
    
      // Read command line arguments.
      for ( int i = 0; i < argc; i++ ) {
        if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-num_steps" ) == 0 ) ) {
            num_steps = atol( argv[ ++i ] );
            printf( "  User num_steps is %ld\n", num_steps );
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Pi Options:\n" );
            printf( "  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
      }
      
// *************** CUDA 
 
// TODO free the h_step_mult

      // Timer products.
      struct timeval begin, end;



    gettimeofday( &begin, NULL );

     

    double *result = (double*)malloc(sizeof(double)); 
    double *d_steps, *d_result; 
    long *d_size, *h_num_steps = &num_steps;
// we have the num of steps	
    *step = 1.0/(double) num_steps;
    *result = 0.0; 
    double *h_steps = (double *)malloc( __NUM_STEPS__ * sizeof(double)); 
    std::fill_n(h_steps,__NUM_STEPS__, 0); 
    printf("the value of the first element of the array before : %lf. \n", h_steps[0]); 
    cudaMalloc((void **)&d_steps,  __NUM_STEPS__ * sizeof(double)); 
    cudaMalloc((void **)&d_size, sizeof(long)); 
    cudaMalloc((void **)&d_result, sizeof(double)); 
    printf("number of steps %ld.\n", *h_num_steps ); 
    printf("number of steps using the variabl num_steps %ld. \n", __NUM_STEPS__);
    cudaMemcpy(h_num_steps, d_size, sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(h_steps, d_steps, __NUM_STEPS__ * sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(result, d_result, sizeof(double), cudaMemcpyHostToDevice); 
    
    //atomic_pi<<<num_steps, 1>>>(d_steps, d_result, d_size); 
    atomic_pi_without_step<<<__NUM_STEPS__, 1>>>(d_steps, d_result); 
    //test<<<num_steps, 1>>>(d_steps, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(result, d_result, sizeof(double), cudaMemcpyDeviceToHost); 
    cudaMemcpy(h_steps, d_steps, __NUM_STEPS__* sizeof(double), cudaMemcpyDeviceToHost); 
         gettimeofday( &end, NULL );
    cudaDeviceSynchronize();
    
    // pi = step * sum 
    *result = (1.0/__NUM_STEPS__) * (*result);
    printf("the first element of the array is : %lf \n", h_steps[0]); 
/*    for(unsigned int i = 0 ; i < num_steps; i++) {
	*result += h_steps[i]; 
    }*/
      // Calculate time.
      double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );
                
      printf("\n pi with %ld steps is %lf in %lf seconds\n ",__NUM_STEPS__,*result,time);
    
    cudaFree(d_size); 
    cudaFree(d_result); 
    cudaFree(d_steps); 
    
    free(result);
    free(step);


}	
/*  for (i=1;i<= num_steps; i++){
		  x = (i-0.5)*step;
		  sum = sum + 4.0/(1.0+x*x);
	  }

	  pi = step * sum;
*/

