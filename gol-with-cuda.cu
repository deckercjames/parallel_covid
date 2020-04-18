#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

#include <cuda.h>
#include <cuda_runtime.h>

extern "C" void covid_allocateMem( unsigned int** infectedCounts,
                        unsigned int** recoveredCounts,
                        unsigned int** infectedCountResults,
                        unsigned int** recoveredCountResults,
                        int numCities){

    int dataLength = numCities * sizeof(unsigned int);

    cudaMallocManaged( infectedCounts, dataLength );
    cudaMallocManaged( recoveredCounts, dataLength );
    cudaMallocManaged( infectedCountResults, dataLength );
    cudaMallocManaged( recoveredCountResults, dataLength );

}

extern "C" void gol_freeMem( unsigned int* infectedCounts,
                        unsigned int* recoveredCounts,
                        unsigned int* infectedCountResults,
                        unsigned int* recoveredCountResult){
    cudaFree(infectedCounts);
    cudaFree(recoveredCounts);
    cudaFree(infectedCountResults);
    cudaFree(recoveredCountResults);
}

static inline void gol_swap( unsigned char **pA, unsigned char **pB)
{
    // You write this function - it should swap the pointers of pA and pB.
    //declare a temp to store A
    unsigned char * temp = *pA;
    //set a to b
    *pA = *pB;
    //set b to the stored val of a
    *pB = temp;
}
 
__device__ static inline unsigned int covid_getProbOfSpread(const unsigned char* data, 
                       int pop1, int infectedCount1, int rank1,
                       int pop1, int infectedCount1, int rank1,
					   double distand) 
{
  
  return 0;
    
}


// Don't modify this function or your submitty autograding may incorrectly grade otherwise correct solutions.
extern "C" void gol_printWorld(unsigned char* data, unsigned int worldSize, int myrank)
{
    int i, j;
    int row;

    //print ghost row
    printf("Gst %2d: ", (myrank*worldSize));
    for( j = 0; j < worldSize; j++){
        printf("%u ", (unsigned int) data[j]);
    }
    printf("\n");

    for( i = 1; i < worldSize + 1; i++)
    {
        row = (myrank * worldSize) + i - 1;
    	printf("Row %2d: ", row);
    	for( j = 0; j < worldSize; j++)
    	{
    	    printf("%u ", (unsigned int) data[(i*worldSize) + j]);
    	}
    	printf("\n");
    }

    printf("Gst %2d: ", ((myrank+1)*worldSize - 1));
    for( j = 0; j < worldSize; j++){
        printf("%u ", (unsigned int) data[((worldSize+1)*worldSize) + j]);
    }

    printf("\n\n");
}



__global__ void covid_intracity_kernel( unsigned int** infectedCounts,
                        unsigned int** recoveredCounts,
                        unsigned int** infectedCountResults,
                        unsigned int** recoveredCountResults,
                        int dataLength)
{

    //Declare variables that will be used
    int infected, recovered;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    while(index < dataLength){

        infected = infectedCounts[index];
        recovered = recoveredCounts[intdex];

        infectedCountResults[index]  = infected * 1.01;
        recoveredCountResults[index] = recovered + infected * 0.1;

        //increment the index
        index += blockDim.x * gridDim.x;

    }
    
}



extern "C" bool covid_itracity_kernelLaunch( unsigned int** infectedCounts,
                        unsigned int** recoveredCounts,
                        unsigned int** infectedCountResults,
                        unsigned int** recoveredCountResults,
                        int dataLength,
                        ushort threadsCount)
{


    //calculate the number of blocks based on the threads per block
    int blockCount = dataLength / threadsCount;

    //run one itterations
    covid_intracity_kernel<<<blockCount, threadsCount>>>( *data, worldWidth, worldHeight, *resultData);

    gol_swap( infectedCountResults, infectedCounts );
    gol_swap( recoveredCountResults, recoveredCounts );

    cudaDeviceSynchronize();

    return true;
}







