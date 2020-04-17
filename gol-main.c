
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

#include <mpi.h>

extern void gol_initMaster( unsigned char** data, unsigned char** resultData, unsigned int pattern,
    size_t worldSize, int myrank, int numranks );

extern void gol_printWorld(unsigned char* data, unsigned int worldSize, int myrank);

extern bool gol_kernelLaunch(unsigned char** data,
                        unsigned char** resultData,
                        size_t worldWidth,
                        size_t worldHeight,
                        ushort threadsCount);

extern void gol_freeMem( unsigned char* data, unsigned char* resultData );



struct City{
    int totalPopulation;
    int lattitule;
    int longitue;
    int rank;
    double socialDistancingFactor;
}



int main(int argc, char *argv[])
{

    //declare variables
    int i;


    //Graph
    unsigned double adjacencyMatrix[ cityCount * cityCount ];


    //declare city array
    struct cities cityData[ cityCount ];

    // Current state of world. 
    unsigned int* infectedCount = 0;
	// Result from last compute of world.
	unsigned int* infectedCountResult = 0;




    //rank data
    int myRank = 0;
    int numRanks = 0;

    //setup mpi rank

    //allocate memory for graph, cityData, infectedCount/Result
    //  in cuda file: cudaMallocManaged()

    //read in city data

    //pass city data to all other ranks

    //create graph

    for(i = 0; i<iterations; i++){

        //inter-city update

        //pass infectedCount of all cities to all other ranks

        //spread of desease

        //pass infectedCount of all cities to all other ranks

    }

    //write results

}



