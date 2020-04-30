
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<string.h>

#include "mpi.h"
#include "covidTypes.h"

//extern struct City;
//extern struct InfectedCity;
const int iterations = 10;
const int threadsCount = 64;

extern void covid_allocateMem( unsigned int** infectedCounts,
                        unsigned int** recoveredCounts,
                        unsigned int** infectedCountResults,
                        unsigned int** recoveredCountResults,
                        int numCities);

extern bool covid_kernelLaunch(struct City** cityData,
                        struct InfectedCity** allReleventInfectedCities,
                        struct InfectedCity** allReleventInfectedCitiesResult,
                        int dataLength,
                        ushort threadsCount,
                        char intracity_or_spread);

extern void covid_freeMem( unsigned int* infectedCounts,
                        unsigned int* recoveredCounts,
                        unsigned int* infectedCountResults,
                        unsigned int* recoveredCountResult);



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

const char filename[50] = "uscities.csv";
const int fileLength = 5309769;

MPI_Datatype getInfectedCityDataType(){

    const int fieldCount = 5;
    int          blocklengths[fieldCount] = {1, 1, 1, 1, 1};
    MPI_Datatype types       [fieldCount] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT};
    MPI_Aint     offsets     [fieldCount];

    //check the offset of each field
    offsets[0] = offsetof(struct InfectedCity, susceptibleCount);
    offsets[1] = offsetof(struct InfectedCity, infectedCount);
    offsets[2] = offsetof(struct InfectedCity, recoveredCount);
    offsets[3] = offsetof(struct InfectedCity, deceasedCount);
    offsets[4] = offsetof(struct InfectedCity, iterationOfInfection);

    //the data type to be created
    MPI_Datatype mpi_infectedCity_type;

    //create and commit the datatype
    MPI_Type_create_struct(fieldCount, blocklengths, offsets, types, &mpi_infectedCity_type);
    MPI_Type_commit(&mpi_infectedCity_type);

    //return the datatype to the main function
    return mpi_infectedCity_type;

}


void MPI_passInfectionData(struct InfectedCity* allReleventInfectedCities,
                            struct InfectedCity** largeCitiesByRank_head, int* largeCitiesByRank_length,
                            int myRank, int numRanks,
                            MPI_Datatype mpi_infectedCity_type,
                            int iteration){
    int i;
    int tag = iteration * 10;

    MPI_Request request0, request1;
    MPI_Status status;

    //setup to recieve data
    for(i = 0; i<numRanks; i++){
        if(i==myRank) continue;
        // Exchange row data with MPI Ranks using MPI_Isend/Irecv.
        //                 data                       length             data type  source   tag        MPI_COMM    MPI_Request
        MPI_Irecv(largeCitiesByRank_head[i], largeCitiesByRank_length[i], MPI_INT,     i,    tag+1,  MPI_COMM_WORLD, &request0);
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
    //struct City releventCityData[ totalReleventCities ];
    struct City* cityData;
    int cityDataLength, numSmallCities;
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
    struct InfectedCity** rankDataPointers = (struct InfectedCity**) malloc(numRanks * sizeof(struct InfectedCity*));
    int* rankLengths = (int*) malloc(numRanks * sizeof(int));


    //allocate memory for cityData, infectedCount/Result
    //  in cuda file: cudaMallocManaged()

    //read in city data
    readFile(filename, fileLength, myRank, numRanks, &cityData, &cityDataLength, &numSmallCities);

    printf("cityDataLen: %d numSmallCities: %d rank: %d\n", cityDataLength, numSmallCities, myRank);
    
    if(myRank == 0){
        for(i = 0; i < cityDataLength; i += 100){
            printf("rank %d city %d population %d density %d lat %lf long %lf name %s state %s cityRanking %d\n", 
            myRank, i, cityData[i].totalPopulation, cityData[i].density, cityData[i].lattitude,
            cityData[i].longitude, cityData[i].cityName, cityData[i].state, cityData[i].cityRanking);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(myRank == 1){
        for(i = 0; i < cityDataLength; i += 100){
            printf("rank %d city %d population %d density %d lat %lf long %lf name %s state %s cityRanking %d\n", 
            myRank, i, cityData[i].totalPopulation, cityData[i].density, cityData[i].lattitude,
            cityData[i].longitude, cityData[i].cityName, cityData[i].state, cityData[i].cityRanking);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);    
    if(myRank == 2){
        for(i = 0; i < cityDataLength; i += 100){
            printf("rank %d city %d population %d density %d lat %lf long %lf name %s state %s cityRanking %d\n", 
            myRank, i, cityData[i].totalPopulation, cityData[i].density, cityData[i].lattitude,
            cityData[i].longitude, cityData[i].cityName, cityData[i].state, cityData[i].cityRanking);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);    
    if(myRank == 3){
        for(i = 0; i < cityDataLength; i += 100){
            printf("rank %d city %d population %d density %d lat %lf long %lf name %s state %s cityRanking %d\n", 
            myRank, i, cityData[i].totalPopulation, cityData[i].density, cityData[i].lattitude,
            cityData[i].longitude, cityData[i].cityName, cityData[i].state, cityData[i].cityRanking);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);    
    if(myRank == 4){
        for(i = 0; i < cityDataLength; i += 100){
            printf("rank %d city %d population %d density %d lat %lf long %lf name %s state %s cityRanking %d\n", 
            myRank, i, cityData[i].totalPopulation, cityData[i].density, cityData[i].lattitude,
            cityData[i].longitude, cityData[i].cityName, cityData[i].state, cityData[i].cityRanking);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(myRank == 5){
        for(i = 0; i < cityDataLength; i += 100){
            printf("rank %d city %d population %d density %d lat %lf long %lf name %s state %s cityRanking %d\n", 
            myRank, i, cityData[i].totalPopulation, cityData[i].density, cityData[i].lattitude,
            cityData[i].longitude, cityData[i].cityName, cityData[i].state, cityData[i].cityRanking);
        }
    }


    //pass city data to all other ranks

    //create adjacency list

    //start infection

    for(i = 0; i<iterations; i++){

        //intra-city update
        covid_kernelLaunch(&cityData,
            &allReleventInfectedCities,
            &allReleventInfectedCitiesResult,
            smallCityCountWithinRank + largeCitiesByRank_length[myRank],
            threadsCount,
            'i');

        //pass infectedCount of all cities to all other ranks
        //MPI_passInfectionData(allInfectedBigCities, rankDataPointers, rankLengths, myRank, numRanks, mpi_infectedCity_type, i);

        //spread of desease
        /*
        covid_kernelLaunch(&cityData,
            allReleventInfectedCities,
            allReleventInfectedCitiesResult,
            smallCityCountWithinRank + allLargeCityCount,
            threadsCount,
            's');
            */

    }

    //write results

}



void readFile(const char* fileName, int numChars, int rank, int numRanks, 
struct City** cityData, int* cityDataLength, int* numSmallCities){
    MPI_File f;
    MPI_Status status;
    char* buf;
    char* endptr;
    int startPos = numChars * ((float) rank / numRanks);
    int endPos = numChars * (((float) rank + 1)/ numRanks);
    int bufSize = 2*numChars/numRanks;
    int i, tokenNum, smallCityIndex = -1, largeCityIndex = -1;
    char token[3000] = {'\0'};
    char tmpChar = '\0';
    static const struct City blankCity;
    struct City tmpCity;
    struct City largeCities[2000];
    char firstState[3] = {'\0'};
    char curState[3] = {'\0'};
    char finalState[3] = {'\0'};
    short firstStateRecorded = 0, onFinalState = 0, finalStateRecorded = 0;
    if(rank == numRanks - 1){
        //ceiling of numChars/numRanks
        bufSize = numChars/numRanks + (numChars % numRanks != 0);
    }

    //we will be reading twice the expected size at each rank
    //because most ranks will have to go over expected size
    buf = (char*) calloc(bufSize, sizeof(char));
    *cityData = (struct City*) calloc(bufSize/180, sizeof(struct City));//approx 180 chars per line
    printf("before file open bufSize: %d startPos: %d\n", bufSize, startPos);
    MPI_File_open(MPI_COMM_WORLD, fileName, MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
    printf("before read at\n");
    MPI_File_read_at(f, startPos, buf, bufSize, MPI_CHAR, &status);
    printf("before main loop\n");

    for(i = 0; i < bufSize; i++){
        if(buf[i] == '\n'){
            if(smallCityIndex == -1){
                smallCityIndex++;
                largeCityIndex++;
            }
            tokenNum = 0;
            token[0] = '\0';
            tmpCity = blankCity;
            continue;
        }
        if(smallCityIndex == -1)
            continue;
        
        if(i == endPos - startPos){
            onFinalState = 1;
        }
        if(i > 2 && buf[i] == '\"' && buf[i-1] == ',' && buf[i-2] == '\"'){//the end of a token is found
            switch(tokenNum){
                case 0://city name
                    strcpy(tmpCity.cityName, token);
                    break;
                case 2://state abrev
                    if(!firstStateRecorded){
                        strcpy(firstState, token);
                        firstStateRecorded = 1;
                    }
                    if(rank == 0){// && strcmp(curState, token) != 0
                        printf("rank: %d, state: %s, smallI: %d, largeI: %d, i: %d\n", 
                        rank, token, smallCityIndex, largeCityIndex, i);
                    }
                    strcpy(curState, token);
                    if(onFinalState && !finalStateRecorded){
                        strcpy(finalState, curState);
                        finalStateRecorded = 1;
                    }
                    if(onFinalState && strcmp(curState, finalState) != 0){
                        i = bufSize;
                        printf("final state: %s cur state: %s\n", finalState, curState);
                    }
                    strcpy(tmpCity.state, token);
                    break;
                case 8://lat
                    tmpCity.lattitude = strtod(token, &endptr);
                    break;
                case 9://long
                    tmpCity.longitude = strtod(token, &endptr);
                    break;
                case 10://pop
                    tmpCity.totalPopulation = atoi(token);
                    break;
                case 11://density
                    tmpCity.density = atoi(token);
                    break;
                case 16://rank
                    tmpCity.cityRanking = atoi(token);
                    if(strcmp(curState, firstState) == 0)
                        break;
                    if(tmpCity.cityRanking < 3){
                        largeCities[largeCityIndex] = tmpCity;
                        largeCityIndex++;
                    }else{
                        (*cityData)[smallCityIndex] = tmpCity;
                        smallCityIndex++;
                    }
                    break;
            }
            tokenNum++;
            token[0] = '\0';
            continue;
        }
        if(buf[i] != '\"' && buf[i] != ','){
            tmpChar = buf[i];
            strcat(token, &tmpChar);
        }
    }
    printf("before second loop\n");
    for(i = smallCityIndex; i < smallCityIndex + largeCityIndex; i++){
        (*cityData)[i] = largeCities[i - smallCityIndex];
    }
    printf("after second loop\n");
    *cityDataLength = smallCityIndex + largeCityIndex;
    *numSmallCities = smallCityIndex;
    free(buf);
}