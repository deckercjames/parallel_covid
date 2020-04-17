#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

#include <cuda.h>
#include <cuda_runtime.h>


extern "C" void gol_initMaster( unsigned char** data, unsigned char** resultData, unsigned int pattern,
    size_t worldSize, int myrank, int numranks )
{

}

extern "C" void gol_freeMem( unsigned char* data, unsigned char* resultData ){
    cudaFree(data);
    cudaFree(resultData);
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



__global__ void gol_kernel(const unsigned char* d_data,
                            unsigned int worldWidth,
                            unsigned int worldHeight,
                            unsigned char* d_resultData){

    //Declare variables that will be used
    size_t y, y0, y1, y2;
    size_t x, x0, x1, x2;
    int aliveCount;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // int stopIndex = index + blockDim.x;

    while(index < worldWidth*worldHeight){

        //calculate x and y from index
        y = index / worldWidth;
        x = index - (y * worldWidth);

        // set: y0, y1 and y2
        y0 = ((y + worldHeight - 1) % worldHeight) * worldWidth;
        y1 = y * worldWidth;
        y2 = ((y + 1) % worldHeight) * worldWidth;

        // set x0, x2, call countAliveCells and compute if g_resultsData[y1 + x] is 0 or 1
        x1 = x;
        x0 = (x1 + worldWidth - 1) % worldWidth;
        x2 = (x1 + 1) % worldWidth;

        //count the alive cells
        aliveCount = gol_countAliveCells( d_data, x0, x1, x2, y0, y1, y2);

        //set the result
        d_resultData[x1+y1] = (d_data[x1+y1] && aliveCount==2) || aliveCount==3;

        //increment the index
        index += blockDim.x * gridDim.x;

    }
    
}



extern "C" bool gol_kernelLaunch(unsigned char** data,
                        unsigned char** resultData,
                        size_t worldWidth,
                        size_t worldHeight,
                        ushort threadsCount)
{


    //calculate the number of blocks based on the threads per block
    int blockCount = ( worldWidth * worldHeight ) / threadsCount;

    //run one itterations
    gol_kernel<<<blockCount, threadsCount>>>( *data, worldWidth, worldHeight, *resultData);
    gol_swap( resultData, data );

    cudaDeviceSynchronize();

    return true;
}







