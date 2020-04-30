#include<stdio.h>
#include<stdlib.h>
#include "mpi.h"
#define LINESIZE 500

extern struct City;

/*struct City{
    int totalPopulation;
    int density;

    int cityRanking;
    double lattitude;
    double longitude;
    char cityName[50];
    char state[2];

    struct City* connectedCitiesIndicies;
    double* edgeWeights;
};*/

//takes the fileName of the csv city dataset, puts all relevant data in cityData,
//creates connections and weights between cities
//order of cityData: [all small cities in this rank, all larges cities in this rank]
//one of us can write a function to pass large cities to all ranks (we will know the
//start index of the large cities from numSmallCities)
//starts reading at numChars*rank/numRanks. All data is read in with one call, and then
//analyzed internally. Once the start of a new state is found, add that state
//to cityData as well as any states up to numLines*(rank+1)/numRanks (finishes adding the last state)
//uscities.csv has 5309769 chars

//*there is a 1 in 50k edge case where a state gets read by two ranks
//but I'm going to ignore it lol*
void readFile(const char* fileName, int numChars, int rank, int numRanks, 
struct City** cityData, int* cityDataLength, int* numSmallCities);