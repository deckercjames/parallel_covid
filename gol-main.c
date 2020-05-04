
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<string.h>

#include "mpi.h"
#include "covidTypes.h"

//extern struct City;
//extern struct InfectedCity;
const int iterations = 1;
const int threadsCount = 64;

extern void covid_allocateMem_CityData(
                        struct City** cityData,
                        int numCitiesWithinRank);

extern void covid_reallocateMem_CityData(
                        struct City** cityData,
                        struct City** cityDataTemp,
                        int numCitiesWithinRank,
                        int numRelevantCities);

extern void covid_allocateMem_InfectedCities_init(
                        struct City** cityData,
                        struct InfectedCity** infectedCities,
                        struct InfectedCity** infectedCitiesResult,
                        int numRelevantCities);

extern bool covid_intracity_kernelLaunch(struct City** cityData,
                        struct InfectedCity** allReleventInfectedCities,
                        struct InfectedCity** allReleventInfectedCitiesResult,
                        int dataLength,
                        ushort threadsCount);

extern bool covid_spread_kernelLaunch(struct City** cityData,
                        struct InfectedCity** allReleventInfectedCities,
                        struct InfectedCity** allReleventInfectedCitiesResult,
                        int mySmallCityCount,
                        int myLargeCityCount,
                        int allLargeCityCount,
                        ushort threadsCount);

extern void covid_freeMem(
                        struct City** cityData,
                        struct InfectedCity** infectedCities,
                        struct InfectedCity** infectedCitiesResult);



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



MPI_Datatype getCityDataDataType();

MPI_Datatype getInfectedCityDataType();


void MPI_passNumberOfLargeCities(int* largeCitiesByRank_length, int myRank, int numRanks){

    int i;

    MPI_Request request0, request1;
    MPI_Status status;

    //receive the number of large cites from all other ranks
    for(i = 0; i<numRanks; i++){
        if(i==myRank) continue;
        // Exchange row data with MPI Ranks using MPI_Isend/Irecv.
        //                  data               length   data type  source  tag    MPI_COMM    MPI_Request
        MPI_Irecv(largeCitiesByRank_length+i,    1,     MPI_INT,     i,    1,  MPI_COMM_WORLD, &request0);
    }

    //send the number of large cities in my rank to all other ranks
    for(i = 0; i<numRanks; i++){
        if(i==myRank) continue;
        //                  data                    length  data type    dest  tag     MPI_COMM     MPI_Request
        MPI_Isend(largeCitiesByRank_length+myRank,    1,    MPI_INT,     i,    1,  MPI_COMM_WORLD, &request1);
    }

    MPI_Wait(&request0, &status);
    MPI_Wait(&request1, &status);

}

void MPI_passLargeCities(struct City** largeCitiesByRank_head,
                            MPI_Datatype mpi_cityData_type,
                            int* largeCitiesByRank_length, 
                            int myRank, int numRanks)
{

    int i;

    MPI_Request request0, request1;
    MPI_Status status;

    //receive the large cites from all other ranks
    for(i = 0; i<numRanks; i++){
        if(i==myRank) continue;
        // Exchange row data with MPI Ranks using MPI_Isend/Irecv.
        //                  data                      length                 data type       source  tag    MPI_COMM     MPI_Request
        MPI_Irecv(largeCitiesByRank_head[i], largeCitiesByRank_length[i], mpi_cityData_type,    i,    1,  MPI_COMM_WORLD, &request0);
    }

    //send the large cities in my rank to all other ranks
    for(i = 0; i<numRanks; i++){
        if(i==myRank) continue;
        //                  data                            length                    data type       dest   tag     MPI_COMM     MPI_Request
        MPI_Isend(largeCitiesByRank_head[myRank], largeCitiesByRank_length[myRank], mpi_cityData_type,  i,    1,  MPI_COMM_WORLD, &request1);
    }

    MPI_Wait(&request0, &status);
    MPI_Wait(&request1, &status);

}

/*
Uses the 'numSmallCities' and numLargeCities for each rank to reallocate the correct amount of memory
for cityData and copy all existing cities to the new array. Also calculated the head of each rank's data
*/
void setupCityData(struct City** cityData, struct City** largeCitiesByRank_head,
                        struct InfectedCity* allReleventInfectedCities, struct InfectedCity** largeInfectedCitiesByRank_head,
                        int numSmallCities, int* largeCitiesByRank_length, int myRank, int numRanks)
{

    int i;

    struct City* cityDataTemp;

    //this is to calculate the headPointers
    int headPointerCounter = numSmallCities;
    int numCitiesWithinRank = numSmallCities + largeCitiesByRank_length[myRank];

    //calculate the total length
    int totalLength = numSmallCities;
    for(i = 0; i < numRanks; i++) totalLength += largeCitiesByRank_length[i];

    //Increase the length of the cityData to hold all Large Cities from other ranks
    // cityData = (struct City*) realloc(cityData, totalLength * sizeof(struct City));
    covid_reallocateMem_CityData(cityData, &cityDataTemp, numCitiesWithinRank, totalLength);

    //calculate the head of each rank's large cities

    //first it the large cities in this rank
    largeCitiesByRank_head[myRank] = *cityData + headPointerCounter;
    largeInfectedCitiesByRank_head[myRank] = allReleventInfectedCities + headPointerCounter;
    headPointerCounter += largeCitiesByRank_length[myRank];
    //next, all other rank's large cities
    for(i = 0; i<numRanks; i++){
        if(i == myRank) continue;//ignore my rank

        //set the head of the large city section for rank i
        largeCitiesByRank_head[i]         = *cityData + headPointerCounter;
        largeInfectedCitiesByRank_head[i] = allReleventInfectedCities + headPointerCounter;
        //increemnt the headPointer counter
        headPointerCounter += largeCitiesByRank_length[i];
    }

}

// void MPI_passLargeCityData(struct City*)


void MPI_passInfectionData(struct InfectedCity** largeCitiesByRank_head, int* largeCitiesByRank_length,
                            int myRank, int numRanks,
                            MPI_Datatype mpi_infectedCity_type,
                            int iteration);


void printLargeCitySample( struct City** largeCitiesByRank_head,
                                struct InfectedCity** largeInfectedCitiesByRank_head,
                                int myRank, int numRanks)
{

    int i;

    if(myRank != 0) return;

    printf("%15s %10s %10s %10s %10s %10s %10s\n", "name", "population", "suseptable", "infected", "recovered", "deseased", "itr");

    for(i = 0; i<numRanks; i++){
        printf("%14s: %10d %10d %10d %10d %10d %10d\n",
            largeCitiesByRank_head[i]->cityName,
            largeCitiesByRank_head[i]->totalPopulation,
            largeInfectedCitiesByRank_head[i]->susceptibleCount,
            largeInfectedCitiesByRank_head[i]->infectedCount,
            largeInfectedCitiesByRank_head[i]->recoveredCount,
            largeInfectedCitiesByRank_head[i]->deceasedCount,
            largeInfectedCitiesByRank_head[i]->iterationOfInfection
        );
    }
}


int main(int argc, char *argv[])
{

    printf("hello world\n");

    //declare variables
    int i, j;

    //Graph

    //declare city array
    //stores city data for all relevent cities (in the same order as 'allReleventInfectedCities')
    //struct City releventCityData[ totalReleventCities ];
    struct City* cityData;
    int cityDataLength, numSmallCities, numLargeCitiesWithinRank, allLargeCityCount, numRelevantCities;
    // Current state of this rank 
    //points to dynamically allocated array of cities
    //Order:
    //[all small cities in this rank, all larges cities in this rank, all large cities in rank b, ...]
    struct InfectedCity* allReleventInfectedCities= 0;
    struct InfectedCity* allReleventInfectedCitiesResult = 0;

    //store where in 
    struct City**         largeCitiesByRank_head;         //[i] = pointer to start of the section of data in 'cityData' for rank i
    struct InfectedCity** largeInfectedCitiesByRank_head; //[i] = pointer to start of the section of data in 'allReleventInfectedCities' for rank i
    int*                  largeCitiesByRank_length;       //[i] = number of big cities in rank i (length to read from head)


    //rank data
    int myRank;
    int numRanks;



    // Setup MPI
    MPI_Init(&argc, &argv);

    //get the data types
    MPI_Datatype mpi_infectedCity_type = getInfectedCityDataType();
    MPI_Datatype mpi_cityData_type = getCityDataDataType();

    //get my rank
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    //read in city data
    readFile(filename, fileLength, myRank, numRanks, &cityData, &cityDataLength, &numSmallCities);

    //set the number of large cities within this rank
    numLargeCitiesWithinRank = cityDataLength - numSmallCities;

    //PASS LARGE CITY DATA TO ALL OTHER RANKS

    largeCitiesByRank_length       = (int*)                  malloc(numRanks * sizeof(int));
    largeCitiesByRank_head         = (struct City**)         malloc(numRanks * sizeof(struct City*));
    largeInfectedCitiesByRank_head = (struct InfectedCity**) malloc(numRanks * sizeof(struct InfectedCity*));

    //first pass the number of large cities to all other ranks so they know how many to recieve
    largeCitiesByRank_length[myRank] = numLargeCitiesWithinRank;
    MPI_passNumberOfLargeCities(largeCitiesByRank_length, myRank, numRanks);

    //calculate the total number of large cities
    allLargeCityCount = 0;
    for(i = 0; i<numRanks; i++) allLargeCityCount += largeCitiesByRank_length[i];
    numRelevantCities = numSmallCities + allLargeCityCount;

    printf("length passed\n");

    MPI_Barrier(MPI_COMM_WORLD);

    //DEBUG LENGTHS
    if(myRank == 0) printf("All lengths for all ranks: \n");
    for(i = 0; i<numRanks; i++){
        MPI_Barrier(MPI_COMM_WORLD);
        if(myRank != i) continue;
        printf("Rank: %2d  totalCitiesInRank: %8d    small: %d    large: [%d] %5d", 
            myRank, (numSmallCities+numLargeCitiesWithinRank), numSmallCities, myRank, numLargeCitiesWithinRank);
        for(j = 0; j<numRanks; j++){
            if(j == myRank) continue;
            printf("     [%d] %5d", j, largeCitiesByRank_length[j]);
        }
        printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //init memory for infected cities
    covid_allocateMem_InfectedCities_init(&cityData, &allReleventInfectedCities, &allReleventInfectedCitiesResult, numRelevantCities);


    //set up cityData now the number of large cities in each rank is known
    setupCityData(&cityData, largeCitiesByRank_head,
        allReleventInfectedCities, largeInfectedCitiesByRank_head,
        numSmallCities, largeCitiesByRank_length, myRank, numRanks);

    MPI_passLargeCities(largeCitiesByRank_head, mpi_cityData_type, largeCitiesByRank_length, myRank, numRanks);

    printf("large cities passed\n");

    MPI_Barrier(MPI_COMM_WORLD);



    //PATIENT ZERO
    if(myRank == 0){
        largeInfectedCitiesByRank_head[myRank]->susceptibleCount -= 1;
        largeInfectedCitiesByRank_head[myRank]->infectedCount += 1;
    }

    //debug
    printLargeCitySample(largeCitiesByRank_head, largeInfectedCitiesByRank_head, myRank, numRanks);

    MPI_Barrier(MPI_COMM_WORLD);

    //INFECTION ITERATIONS

    for(i = 0; i<iterations; i++){

        printf("rank %d, iteration %d\n", myRank, i);

        //intra-city update
        // covid_intracity_kernelLaunch(&cityData,
        //     &allReleventInfectedCities,
        //     &allReleventInfectedCitiesResult,
        //     cityDataLength,
        //     threadsCount);

        MPI_Barrier(MPI_COMM_WORLD);

        //debug
        printLargeCitySample(largeCitiesByRank_head, largeInfectedCitiesByRank_head, myRank, numRanks);

        //pass infectedCount of all cities to all other ranks
        // MPI_passInfectionData(largeInfectedCitiesByRank_head, largeCitiesByRank_length, myRank, numRanks, mpi_infectedCity_type, i);

        //spread of desease
        // covid_spread_kernelLaunch(&cityData,
        //     &allReleventInfectedCities,
        //     &allReleventInfectedCitiesResult,
        //     numSmallCities,
        //     numLargeCitiesWithinRank,
        //     allLargeCityCount,
        //     threadsCount);

    }
    


    //write results


    //finalize
    MPI_Finalize();

    //free memory
    covid_freeMem(&cityData, &allReleventInfectedCities, &allReleventInfectedCitiesResult);
    free(largeCitiesByRank_length);
    free(largeCitiesByRank_head);
    free(largeInfectedCitiesByRank_head);


}

MPI_Datatype getCityDataDataType()
{

    const int fieldCount = 7;
    int blocklengths [fieldCount] = {50, 2, 1, 1, 1, 1, 1};
    MPI_Datatype types [fieldCount] = 
    {
        MPI_CHAR, MPI_CHAR,
        MPI_INT, MPI_INT, MPI_INT,
        MPI_DOUBLE, MPI_DOUBLE
    };

    //check the offset of each field
    MPI_Aint offsets [fieldCount];
    offsets[0] = offsetof(struct City, cityName);
    offsets[1] = offsetof(struct City, state);
    offsets[2] = offsetof(struct City, cityRanking);
    offsets[3] = offsetof(struct City, totalPopulation);
    offsets[4] = offsetof(struct City, density);
    offsets[5] = offsetof(struct City, lattitude);
    offsets[6] = offsetof(struct City, longitude);

    //the data type to be created
    MPI_Datatype mpi_cityData_type;

    //create and commit the datatype
    MPI_Type_create_struct(fieldCount, blocklengths, offsets, types, &mpi_cityData_type);
    MPI_Type_commit(&mpi_cityData_type);

    //return the datatype to the main function
    return mpi_cityData_type;
}


MPI_Datatype getInfectedCityDataType()
{
    const int fieldCount = 5;
    
    MPI_Datatype mpi_infectedCity_type;
 
    //create and commit the datatype
    MPI_Type_contiguous( fieldCount, MPI_INT, &mpi_infectedCity_type );
    MPI_Type_commit(&mpi_infectedCity_type);

    //return the datatype to the main function
    return mpi_infectedCity_type;
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

    // *cityData = (struct City*) calloc(bufSize/180, sizeof(struct City));//approx 180 chars per line
    covid_allocateMem_CityData(cityData, bufSize/180);

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





void MPI_passInfectionData(struct InfectedCity** largeInfectedCitiesByRank_head, int* largeCitiesByRank_length,
                            int myRank, int numRanks,
                            MPI_Datatype mpi_infectedCity_type,
                            int iteration)
{
    int i;
    int tag = iteration * 10;

    MPI_Request request0, request1;
    MPI_Status status;

    //setup to recieve data
    for(i = 0; i<numRanks; i++){
        if(i==myRank) continue;
        // Exchange row data with MPI Ranks using MPI_Isend/Irecv.
        //                 data                               length             data type  source   tag        MPI_COMM    MPI_Request
        MPI_Irecv(largeInfectedCitiesByRank_head[i], largeCitiesByRank_length[i], MPI_INT,     i,    tag + i,  MPI_COMM_WORLD, &request0);
    }

    //send data
    for(i = 0; i<numRanks; i++){
        if(i==myRank) continue;
        //                 data                              length                   data type          dest     tag       MPI_COMM     MPI_Request
        MPI_Isend(largeInfectedCitiesByRank_head[i], largeCitiesByRank_length[i], mpi_infectedCity_type,   i,   tag + i,  MPI_COMM_WORLD, &request1);
    }

    MPI_Wait(&request0, &status);
    MPI_Wait(&request1, &status);
}




