
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

#include <mpi.h>

extern void covid_allocateMem( unsigned int** infectedCounts,
                        unsigned int** recoveredCounts,
                        unsigned int** infectedCountResults,
                        unsigned int** recoveredCountResults,
                        int numCities);

extern void covid_itracity_kernelLaunch(unsigned int** infectedCounts,
                        unsigned int** recoveredCounts,
                        unsigned int** infectedCountResults,
                        unsigned int** recoveredCountResults,
                        int dataLength,
                        ushort threadsCount);

extern bool covid_spread_kernelLaunch(unsigned int** infectedCounts,
                        unsigned int** recoveredCounts,
                        unsigned int** infectedCountResults,
                        unsigned int** recoveredCountResults,
                        int dataLength,
                        ushort threadsCount);

extern void covid_freeMem( unsigned int* infectedCounts,
                        unsigned int* recoveredCounts,
                        unsigned int* infectedCountResults,
                        unsigned int* recoveredCountResult);


extern struct City;
extern struct InfectedCity;

MPI_Datatype getInfectedCityDataType(){

    /* create a type for struct car */
    const int fieldCount = 3;
    int          blocklengths[fieldCount] = {1, 1, 1};
    MPI_Datatype types       [fieldCount] = {MPI_INT, MPI_INT};
    MPI_Aint     offsets     [fieldCount];

    MPI_Datatype mpi_infectedCity_type;

    offsets[0] = offsetof(InfectedCity, infectedCount);
    offsets[1] = offsetof(InfectedCity, recoveredCount);
    offsets[2] = offsetof(InfectedCity, iterationOfInfection);

    MPI_Type_create_struct(fieldCount, blocklengths, offsets, types, &mpi_infectedCity_type);
    MPI_Type_commit(&mpi_infectedCity_type);

    return mpi_infectedCity_type;

}


void MPI_passInfectionData(struct InfectedCity* allReleventInfectedCities,
                            struct InfectedCity** largeCitiesByRank_head, int* largeCitiesByRank_length,
                            int myRank, int numRanks,
                            MPI_Datatype mpi_infectedCity_type,
                            int iteration)
{

    int i;

    MPI_Request request0, request1;
    MPI_Status status;

    //setup to recieve data
    for(i = 0; i<numRanks; i++){
        if(i==myRank) continue;
        // Exchange row data with MPI Ranks using MPI_Isend/Irecv.
        //                 data                       length                     data type       source    tag       MPI_COMM      MPI_Request
        MPI_Irecv(largeCitiesByRank_head[i], largeCitiesByRank_length[i], mpi_infectedCity_type,   i,   iteration,  MPI_COMM_WORLD, &request0);
    }

    //send data
    for(i = 0; i<numRanks; i++){
        if(i==myRank) continue;
        //                 data                       length                   data type          dest     tag          MPI_COMM     MPI_Request
        MPI_Isend(largeCitiesByRank_head[i], largeCitiesByRank_length[i], mpi_infectedCity_type,   i,   iteration,  MPI_COMM_WORLD, &request1);
    }

    MPI_Wait(&request0, &status);
    MPI_Wait(&request1, &status);

}



int main(int argc, char *argv[])
{

    //declare variables
    int i;

    int smallCityCountWithinRank;
    int allLargeCityCount;
    int totalReleventCities = smallCityCountWithinRank + allLargeCityCount;


    //Graph

    //declare city array
    //stores city data for all relevent cities (in the same order as 'allReleventInfectedCities')
    struct City releventCityData[ totalReleventCities ];

    // Current state of this rank 
    //points to dynamically allocated array of cities
    //Order:
    //[all small cities in this rank, all larges cities in this rank, all large cities in rank b, ...]
    struct InfectedCity* allReleventInfectedCities= 0;
    struct InfectedCity* allReleventInfectedCitiesResult = 0;

    //store where in 
    struct InfectedCity** largeCitiesByRank_head; //rankDataPointers[i] = pointer to start of the section of data in 'allReleventInfectedCities' for rank i
    int* largeCitiesByRank_length; //rankLength[i] = number of big cities in rank i (length to read from head of rankDataPointers)


    //rank data
    int myRank;
    int numRanks;


    // Setup MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    //get the data type
    MPI_Datatype mpi_infectedCity_type = getInfectedCityDataType();


    //allocate memory for mpi passing pointer
    rankDataPointers = (InfectedCity**) malloc(numRanks * sizeof(InfectedCity*));
    rankLengths = (int*) malloc(numRanks * sizeof(int));


    //allocate memory for cityData, infectedCount/Result
    //  in cuda file: cudaMallocManaged()

    //read in city data

    //pass city data to all other ranks

    //create adjacency list

    //start infection

    for(i = 0; i<iterations; i++){

        //intra-city update
        covid_kernelLaunch(cityData,
            allReleventInfectedCities,
            allReleventInfectedCitiesResult,
            smallCityCountWithinRank + largeCitiesByRank_length[myRank],
            threadsCount)

        //pass infectedCount of all cities to all other ranks
        MPI_passInfectionData(allInfectedBigCities, rankDataPointers, rankLengths, myRank, numRanks, mpi_infectedCity_type, i);

        //spread of desease
        covid_kernelLaunch(cityData,
            allReleventInfectedCities,
            allReleventInfectedCitiesResult,
            smallCityCountWithinRank + allLargeCityCount,
            threadsCount)

    }

    //write results

}



