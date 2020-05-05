#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<string.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "covidTypes.h"

#include <curand.h>
#include <curand_kernel.h>

#define pi 3.14159265358979323846

typedef unsigned long long ticks;


//array used to determine edge weights (along with distance)
//      target city rank: 1  2  3  4  5
//spreading city rank: 1
//                     2
//                     3
//                     4
//                     5
__device__ const double probabilityMultipliers[5][5] = 
{{1, 0.9, 0.8, 0.7, 0.6},
{0.8, 0.7, 0.6, 0.5, 0.4},
{0.7, 0.6, 0.5, 0.4, 0.3},
{0.6, 0.5, 0.4, 0.3, 0.2},
{0.5, 0.4, 0.3, 0.2, 0.1}};

__device__ const double maxSpreadDistances[5][5] = 
{{4000, 1000, 200, 100, 50},
{1000, 200, 100, 50, 25},
{200, 100, 50, 25, 25},
{100, 50, 25, 25, 25},
{50, 25, 25, 25, 25}};

//probability of infection will be m/log(dist)
//where m is the probability multiplier and log
//has the base spreadLogBase
const double spreadLogBase = 10;

__device__ double deg2rad(double);
__device__ double rad2deg(double);

__device__ char * my_strcpy(char *dest, const char *src){
    int i = 0;
    do {
      dest[i] = src[i];
    }
    while (src[i++] != '\0');
    return dest;
}

//only returns 1 (not the same) or 0 (the same)
__device__ int my_strcmp(char *str1, const char *str2){
    int i = 0;
    do {
        if(str1[i] != str2[i]) return 1;
    }
    while (str1[i++] != '\0');
    if(str1[i] != str2[i])return 1;
    return 0;
}

//returns distance between two points (lat and long) in miles
__device__ double coor2distance(double lat1, double lon1, double lat2, double lon2) {
  double theta, dist;
  if ((lat1 == lat2) && (lon1 == lon2)) {
    return 0;
  }
  else {
    theta = lon1 - lon2;
    dist = sin(deg2rad(lat1)) * sin(deg2rad(lat2)) + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * cos(deg2rad(theta));
    dist = acos(dist);
    dist = rad2deg(dist);
    dist = dist * 60 * 1.1515;
    return (dist);
  }
}

__device__ double deg2rad(double deg) {
  return (deg * pi / 180);
}
__device__ double rad2deg(double rad) {
  return (rad * 180 / pi);
}

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
        //initilize all fields for infected cities
        (*infectedCities)[i].susceptibleCount = (*cityData)[i].totalPopulation;
        (*infectedCities)[i].infectedCount  = 0;
        (*infectedCities)[i].recoveredCount = 0;
        (*infectedCities)[i].deceasedCount  = 0;
        (*infectedCities)[i].iterationOfInfection  = -1;
        //the only result field that needs to be initilized  is iteratonOfInfection because it is not recalculated each iteration	
        (*infectedCitiesResult)[i].iterationOfInfection  = -1;
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


__device__ double getDistance(struct City* city1, struct City* city2){
    return coor2distance(city1->lattitude, city1->longitude, city2->lattitude, city2->longitude);
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
        if(newInfections > city->susceptibleCount) newInfections = city->susceptibleCount;
        //new deaths
        newDeaths = (int) (mortalityRate * (*city).infectedCount);

        //new recoveries
        newRecoveries = (int) (recoveryRate * (*city).infectedCount);
        if(newRecoveries == 0 && newDeaths == 0 && (*city).susceptibleCount == 0 && (*city).infectedCount != 0) newRecoveries = 1;
        if(newRecoveries > city->infectedCount) newRecoveries = city->infectedCount;
        if(cityData[index].totalPopulation == 2629150){
            // printf("index %d\n", index);
            printf("Brooklyn newInfections %d\n", newInfections);
            printf("Brooklyn newRecoveries %d\n", newRecoveries);
        }

        //Calculated city results
        (*cityResult).susceptibleCount = (*city).susceptibleCount - newInfections;
        (*cityResult).infectedCount    = (*city).infectedCount + newInfections - newRecoveries - newDeaths;
        (*cityResult).recoveredCount   = (*city).recoveredCount + newRecoveries;
        (*cityResult).deceasedCount    = (*city).deceasedCount + newDeaths;

        //increment the index
        index += blockDim.x * gridDim.x;

    }
    
}








static inline void covid_swap( struct InfectedCity **pA, struct InfectedCity **pB)
{
    // You write this function - it should swap the pointers of pA and pB.
    //declare a temp to store A
    struct InfectedCity * temp = *pA;
    //set a to b
    *pA = *pB;
    //set b to the stored val of a
    *pB = temp;
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
    
    cudaDeviceSynchronize();

    covid_swap( allReleventInfectedCities, allReleventInfectedCitiesResult );

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
    int blockCount = (mySmallCityCount + myLargeCityCount + allLargeCityCount) / threadsCount;

    //run one itterations
    covid_spread_kernel<<<blockCount, threadsCount>>>( *cityData, *allReleventInfectedCities, *allReleventInfectedCitiesResult,
        mySmallCityCount, myLargeCityCount, allLargeCityCount);

    cudaDeviceSynchronize();

    covid_swap( allReleventInfectedCities, allReleventInfectedCitiesResult );

    return true;
}




