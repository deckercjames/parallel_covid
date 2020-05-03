#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<string.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "covidTypes.h"

#include <curand.h>
#include <curand_kernel.h>


//array used to determine edge weights (along with distance)
//      target city rank: 1  2  3  4  5
//spreading city rank: 1
//                     2
//                     3
//                     4
//                     5
const double edgeWeightMultipliers[5][5] = 
{{1, 0.9, 0.8, 0.7, 0.6},
{0.8, 0.7, 0.6, 0.5, 0.4},
{0.7, 0.6, 0.5, 0.4, 0.3},
{0.6, 0.5, 0.4, 0.3, 0.2},
{0.5, 0.4, 0.3, 0.2, 0.1}};

//edges with weights below this won't be recorded
const double minWeight = 0.01;



extern "C" void covid_allocateMem_CityData(
                        struct City** cityData,
                        int numCitiesWithinRank){

    int cityDataLength = numCitiesWithinRank * sizeof(struct City);

    cudaMallocManaged( cityData, cityDataLength );

}

/*
allocates new memory for cityData of length 'numRelevantCities'.
Coppies all existing city data for 'numCitiesWithinRank' to the new memory
*/
extern "C" void covid_reallocateMem_CityData(
                        struct City** cityData,
                        struct City** cityDataTemp,
                        int numCitiesWithinRank,
                        int numRelevantCities){

    int i;

    //calculate the new length for city data
    int newCityDataLength = numRelevantCities * sizeof(struct City);

    //allocate new memory
    cudaMallocManaged( cityDataTemp, newCityDataLength );

    //copy existing cities into new memory
    for(i = 0; i<numCitiesWithinRank; i++){
        (*cityDataTemp)[i] = (*cityData)[i];
    }

    //free old city data
    cudaFree(cityData);

    //set cityData to the new city data
    *cityData = *cityDataTemp;

}

/*
allocates cuda memory for infected cities and their results. Initilizes all infected cities
to have a suseptable population equal to their population, and no infected/recovered/dead people
*/
extern "C" void covid_allocateMem_InfectedCities_init(
                        struct City** cityData,
                        struct InfectedCity** infectedCities,
                        struct InfectedCity** infectedCitiesResult,
                        int numRelevantCities){
    int i;

    int infectedCityDataLength = numRelevantCities * sizeof(struct InfectedCity);

    cudaMallocManaged( infectedCities,       infectedCityDataLength );
    cudaMallocManaged( infectedCitiesResult, infectedCityDataLength );

    //init infected cities
    for(i = 0; i < numRelevantCities; i++){
        (*infectedCities)[i].susceptibleCount = (*cityData)[i].totalPopulation;
        (*infectedCities)[i].infectedCount  = 0;
        (*infectedCities)[i].recoveredCount = 0;
        (*infectedCities)[i].deceasedCount  = 0;
    }

}

extern "C" void covid_freeMem(
                        struct City** cityData,
                        struct InfectedCity** infectedCities,
                        struct InfectedCity** infectedCitiesResult){
    cudaFree(cityData);
    cudaFree(infectedCities);
    cudaFree(infectedCitiesResult);
}

static inline void pointer_swap( struct InfectedCity **pA, struct InfectedCity **pB)
{
    // You write this function - it should swap the pointers of pA and pB.
    //declare a temp to store A
    struct InfectedCity * temp = *pA;
    //set a to b
    *pA = *pB;
    //set b to the stored val of a
    *pB = temp;
}


__global__ void covid_intracity_kernel(
                        struct City* cityData,
                        InfectedCity* allReleventInfectedCities,
                        InfectedCity* allReleventInfectedCitiesResult,
                        int dataLength)
{

    //parameters for SIR model
    //TODO make these arguements when running the program
    double spreadRate = 2.2;
    double infectionDuration = 12.7;
    double recoveryRate = 1 / infectionDuration;
    double mortalityRate = 0.005;

    //Declare variables that will be used
    int newInfections, newRecoveries, newDeaths;
    
    struct InfectedCity* city;
    struct InfectedCity* cityResult;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    while(index < dataLength){

        //get pointers to the indexed city
        city = allReleventInfectedCities + index;
        cityResult = allReleventInfectedCitiesResult + index;

        //SIR Model
        //new infections
        newInfections = (int) (spreadRate * (*city).susceptibleCount * (*city).infectedCount / cityData[index].totalPopulation);
        //if there are both infected and suseptable people, garenty someone will get infected
        if((*city).infectedCount > 0 && (*city).susceptibleCount > 0 && newInfections == 0) newInfections = 1;

        //new deaths
        newDeaths = (int) (mortalityRate * (*city).infectedCount);

        //new recoveries
        newRecoveries = (int) (recoveryRate * (*city).infectedCount);
        if(newRecoveries == 0 && newDeaths == 0 && (*city).susceptibleCount == 0 && (*city).infectedCount != 0) newRecoveries = 1;


        //Calculated city results
        (*cityResult).susceptibleCount = (*city).susceptibleCount - newInfections;
        (*cityResult).infectedCount    = (*city).infectedCount + newInfections - newRecoveries - newDeaths;
        (*cityResult).recoveredCount   = (*city).recoveredCount + newRecoveries;
        (*cityResult).deceasedCount    = (*city).deceasedCount + newDeaths;

        //increment the index
        index += blockDim.x * gridDim.x;

    }
    
}




__global__ void covid_spread_kernel(
                        struct City* cityData,
                        struct InfectedCity* allReleventInfectedCities,
                        struct InfectedCity* allReleventInfectedCitiesResult,
                        int mySmallCityCount,
                        int myLargeCityCount,
                        int allLargeCityCount)
{
    /*
    All of myLargeCities can be infected by any large city
    allLargeInfectedCities[0 .. myLargeCityCount-1] are my large cities
    allLargeInfectedCities[myLargeCityCount .. allLargeCityCount] are other large cities
    */

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int cityIndex;
    int j;

    double probablility;

    //random probability generator
    double rd;
    curandState curand_state;
    curand_init(1235, index, 0, &curand_state);//(seed, sequence, offset, state)


    while(index < myLargeCityCount){

        //the index of this city is the offset of the small cities at the begening of the array
        cityIndex = index + mySmallCityCount;

        //only try to infect city[cityIndex] if it doesn't have any infections yet
        if(allReleventInfectedCities[cityIndex].infectedCount == 0){

            //all cities can infect city[cityIndex]
            for(j = mySmallCityCount; j<mySmallCityCount+allLargeCityCount; j++){

                //probability that city[j] will infect city[cityIndex]
                probablility = 0;

                //TODO write probability function

                //the city at [cityIndex] gets infected
                rd = curand_uniform(&curand_state);
                if(rd < probablility){
                    allReleventInfectedCitiesResult[cityIndex].susceptibleCount = allReleventInfectedCities[cityIndex].susceptibleCount - 1;
                    allReleventInfectedCitiesResult[cityIndex].infectedCount = 1;
                }
            }
        }

        //increment the index
        index += blockDim.x * gridDim.x;

    }
    
}

extern "C" bool covid_intracity_kernelLaunch(struct City** cityData,
                        struct InfectedCity** allReleventInfectedCities,
                        struct InfectedCity** allReleventInfectedCitiesResult,
                        int dataLength,
                        ushort threadsCount)
{


    //calculate the number of blocks based on the threads per block
    int blockCount = dataLength / threadsCount;

    //run one itterations
    covid_intracity_kernel<<<blockCount, threadsCount>>>( *cityData, *allReleventInfectedCities, *allReleventInfectedCitiesResult, dataLength);

    pointer_swap( allReleventInfectedCities, allReleventInfectedCitiesResult );

    cudaDeviceSynchronize();

    return true;
}

extern "C" bool covid_spread_kernelLaunch(struct City** cityData,
                        struct InfectedCity** allReleventInfectedCities,
                        struct InfectedCity** allReleventInfectedCitiesResult,
                        int mySmallCityCount,
                        int myLargeCityCount,
                        int allLargeCityCount,
                        ushort threadsCount)
{


    //calculate the number of blocks based on the threads per block
    int blockCount = myLargeCityCount / threadsCount;

    //run one itterations
    covid_spread_kernel<<<blockCount, threadsCount>>>( *cityData, *allReleventInfectedCities, *allReleventInfectedCitiesResult,
        mySmallCityCount, myLargeCityCount, allLargeCityCount);

    pointer_swap( allReleventInfectedCities, allReleventInfectedCitiesResult );

    cudaDeviceSynchronize();

    return true;
}

__global__ void smallCityConnectionsKernel(struct City** cityData, int dataLength){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    while(index < dataLength){
        
        for(i = 0; i < dataLength; i++){

        }

        index += blockDim.x * gridDim.x;

    }

}

//this should be called before spreading large cities to all ranks
extern "C" void smallCityConnectionsKernelLaunch(struct City** cityData, int dataLength, int threadCount){
    int blockCount = dataLength/threadCount;


}