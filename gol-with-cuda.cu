#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

#include <cuda.h>
#include <cuda_runtime.h>


typedef struct City City;
extern "C" struct City{
    int totalPopulation;
    int density;

    int cityRanking;
    int lattitude;
    int longitude;
    char cityName[50];
    char state[2];

    struct City* connectedCitiesIndicies[];
    double edgeWeights[];
};


extern "C" struct InfectedCity{
    int susceptibleCount;
    int infectedCount;
    int recoveredCount;
    int iterationOfInfection;
};



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

static inline void pointer_swap( unsigned char **pA, unsigned char **pB)
{
    // You write this function - it should swap the pointers of pA and pB.
    //declare a temp to store A
    unsigned char * temp = *pA;
    //set a to b
    *pA = *pB;
    //set b to the stored val of a
    *pB = temp;
}


__global__ void covid_intracity_kernel(
                        City* cityData,
                        InfectedCity* allReleventInfectedCities,
                        InfectedCity* allReleventInfectedCitiesResult,
                        int dataLength)
{

    //parameters for SIR model
    double spreadRate = 2.2;
    double infectionDuration = 12.7;
    double recoveryRate = 1 / infectionDuration;

    //Declare variables that will be used
    int newInfections, newRecoveries;
    

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    while(index < dataLength){

        //get pointers to the indexed city
        City* city = allReleventInfectedCities + index;
        City* cityResult = allReleventInfectedCitiesResult + index;

        //SIR Model
        //new infections
        newInfections = (int) (spreadRate * city.susceptibleCount * city.infectedCount / cityData[index].totalPopulation);
        if(newInfections == 0 && city.susceptibleCount > 0) newInfections = 1;

        //new recoveries
        newRecoveries = (int) (recoveryRate * city.infectedCount);
        if(newRecoveries == 0 && city.susceptibleCount == 0 && city.infectedCount != 0) newRecoveries = 1;

        //Calculated city results
        cityResult.susceptibleCount = city.susceptibleCount - newInfections;
        cityResult.infectedCount    = city.infectedCount + newInfections - newRecoveries;
        cityResult.recoveredCount   = city.recoveredCount + newRecoveries;

        //increment the index
        index += blockDim.x * gridDim.x;

    }
    
}




__global__ void covid_spread_kernel(
                        City** cityData,
                        InfectedCity** allReleventInfectedCities,
                        InfectedCity** allReleventInfectedCitiesResult,
                        int dataLength)
{

    //Declare variables that will be used
    int infected, recovered;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    while(index < dataLength){

        //TODO

        //increment the index
        index += blockDim.x * gridDim.x;

    }
    
}



extern "C" bool covid_kernelLaunch( City** cityData,
                        InfectedCity** allReleventInfectedCities,
                        InfectedCity** allReleventInfectedCitiesResult,
                        int dataLength,
                        ushort threadsCount,
                        char intracity_or_spread)
{


    //calculate the number of blocks based on the threads per block
    int blockCount = dataLength / threadsCount;

    //run one itterations
    if(intracity_or_spread == 'i')
        covid_intracity_kernel<<<blockCount, threadsCount>>>( *cityData, *allReleventInfectedCities, *allReleventInfectedCitiesResult, dataLength);
    else if(intracity_or_spread == 's')
        covid_spread_kernel<<<blockCount, threadsCount>>>( *cityData, *allReleventInfectedCities, *allReleventInfectedCitiesResult, dataLength);

    pointer_swap( allReleventInfectedCities, allReleventInfectedCitiesResult );

    cudaDeviceSynchronize();

    return true;
}







