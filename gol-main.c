
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

extern void covid_printWorld(unsigned char* data, unsigned int worldSize, int myrank);

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


typedef struct City City;
struct City{
    int totalPopulation;
    int density;

    int cityRanking;
    int lattitule;
    int longitue;

    struct City* connectedCitiesIndiceis[];
    double edgeWeights[];
};


struct InfectedCity{
    int infectedCount;
    int recoveredCount;
    int iterationOfInfection;
};


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


void MPI_passInfectionData(struct InfectedCity* allInfectedBigCities,
                            struct InfectedCity** rankDataPointers, int* rankLengths,
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
        //                data             length             data type       source    tag       MPI_COMM      MPI_Request
        MPI_Irecv(rankDataPointers[i], rankLengths[i], mpi_infectedCity_type,   i,   iteration,  MPI_COMM_WORLD, &request0);
    }

    //send data
    for(i = 0; i<numRanks; i++){
        if(i==myRank) continue;
        //               data               length         data type          dest     tag          MPI_COMM     MPI_Request
        MPI_Isend(rankDataPointers[i], rankLengths[i], mpi_infectedCity_type,   i,   iteration,  MPI_COMM_WORLD, &request1);
    }

    MPI_Wait(&request0, &status);
    MPI_Wait(&request1, &status);

}



int main(int argc, char *argv[])
{

    //declare variables
    int i;

    int dataLength;


    //Graph
    // unsigned double adjacencyMatrix[ cityCountPerRank  ];


    //declare city array
    struct cities cityData[ cityCount ];

    // Current state of this rank 
    struct InfectedCity* infectedSmallCitiesWithinRank = 0; //length = number of level 3-5 cities in this rank
    struct InfectedCity* infectedSmallCitiesWithinRankResult = 0;

    //holds infectiondata for all connected cities in the simulation
    //
    struct InfectedCity* allInfectedBigCities = 0;

    struct InfectedCity** rankDataPointers; //rankDataPointers[i] = pointer to start of the section of data in 'allInfectedBigCities' for rank i
    int* rankLengths; //rankLength[i] = number of big cities in rank i (length to read from head of rankDataPointers)


    //rank data
    int myRank = 0;
    int numRanks = 0;


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

        //pass infectedCount of all cities to all other ranks
        MPI_passInfectionData(allInfectedBigCities, rankDataPointers, rankLengths, myRank, numRanks, mpi_infectedCity_type, i);

        //spread of desease

    }

    //write results

}



