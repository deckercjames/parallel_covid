
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<string.h>

#include "mpi.h"
#include "covidTypes.h"



typedef unsigned long long ticks;


//extern struct City;
//extern struct InfectedCity;
const int iterations = 5;
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
void readFile(const char* fileName, int numChars, int rank, int numRanks, 
struct City** cityData, int* cityDataLength, int* numSmallCities);

//takes in list of fileNames, cityData, infectedCityData
//and outputs cityName, state abbrev, city ranking, total population, susceptibleCount, infectedCount
//recoveredCount, deceasedCount in csv format
double outputData(int numFiles, char*** fileNames, int rank, int numRanks, struct City** cityData, 
struct InfectedCity** infectedCityData, int numCurRankCities);

const char filename[50] = "uscities.csv";
const int fileLength = 5309769;

static __inline__ ticks getticks(void)
{
    unsigned int tbl, tbu0, tbu1;
    do {
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
        __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
    } while (tbu0 != tbu1);
    return ((((unsigned long long)tbu0) << 32) | tbl);
}


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
                        struct InfectedCity* allReleventInfectedCitiesResult, struct InfectedCity** largeInfectedCitiesResultByRank_head,
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
    largeInfectedCitiesResultByRank_head[myRank] = allReleventInfectedCitiesResult + headPointerCounter;
    headPointerCounter += largeCitiesByRank_length[myRank];
    //next, all other rank's large cities
    for(i = 0; i<numRanks; i++){
        if(i == myRank) continue;//ignore my rank

        //set the head of the large city section for rank i
        largeCitiesByRank_head[i]         = *cityData + headPointerCounter;
        largeInfectedCitiesByRank_head[i] = allReleventInfectedCities + headPointerCounter;
        largeInfectedCitiesResultByRank_head[i] = allReleventInfectedCitiesResult + headPointerCounter;
        //increemnt the headPointer counter
        headPointerCounter += largeCitiesByRank_length[i];
    }

}

// void MPI_passLargeCityData(struct City*)


void MPI_passInfectionData(struct InfectedCity** largeCitiesByRank_head, int* largeCitiesByRank_length,
                            int myRank, int numRanks,
                            MPI_Datatype mpi_infectedCity_type,
                            int iteration);


void data_result_swap(struct InfectedCity** a1, struct InfectedCity** b1,
                        struct InfectedCity*** a2, struct InfectedCity*** b2)
{
    struct InfectedCity* temp1 = *a1;
    struct InfectedCity** temp2 = *a2;

    *a1 = *b1;
    *b1 = temp1;

    *a2 = *b2;
    *b2 = temp2;
}

void printLargeCitySample( struct City** largeCitiesByRank_head,
                                struct InfectedCity** largeInfectedCitiesByRank_head,
                                int myRank, int numRanks)
{

    int i;

    if(myRank != 0) return;

    printf("%19s %10s %10s %10s %10s %10s %10s\n", "name", "population", "suseptable", "infected", "recovered", "deseased", "itr");

    for(i = 0; i<numRanks; i++){
        printf("%14s, %2s: %10d %10d %10d %10d %10d %10d\n",
            largeCitiesByRank_head[i]->cityName,
            largeCitiesByRank_head[i]->state,
            largeCitiesByRank_head[i]->totalPopulation,
            largeInfectedCitiesByRank_head[i]->susceptibleCount,
            largeInfectedCitiesByRank_head[i]->infectedCount,
            largeInfectedCitiesByRank_head[i]->recoveredCount,
            largeInfectedCitiesByRank_head[i]->deceasedCount,
            largeInfectedCitiesByRank_head[i]->iterationOfInfection
        );
    }
}

void printNYCities(struct City* cityData, struct InfectedCity* infectedData, int myCityCount, int myRank){

    int i;

    if(myRank != 1) return;

    printf("rank: %d %19s %10s %10s %10s %10s %10s %10s\n", myRank, "name", "population", "suseptable", "infected", "recovered", "deseased", "itr");

    for(i = 0; i<myCityCount; i++){

        if(strcmp(cityData[i].state, "NY") != 0) continue;

        if(cityData[i].cityName[0] != 'B') continue;
        if(cityData[i].cityName[1] != 'r') continue;

        printf("rank: %d %14.14s, %2s: %10d %10d %10d %10d %10d %10d",
            myRank,
            cityData[i].cityName,
            cityData[i].state,
            cityData[i].totalPopulation,
            infectedData[i].susceptibleCount,
            infectedData[i].infectedCount,
            infectedData[i].recoveredCount,
            infectedData[i].deceasedCount,
            infectedData[i].iterationOfInfection
        );

        if(strcmp(cityData[i].cityName, "Brooklyn") == 0) printf("<=====");
        printf("\n");
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
    struct City**         largeCitiesByRank_head;               //[i] = pointer to start of the section of data in 'cityData' for rank i
    struct InfectedCity** largeInfectedCitiesByRank_head;       //[i] = pointer to start of the section of data in 'allReleventInfectedCities' for rank i
    struct InfectedCity** largeInfectedCitiesResultByRank_head; //[i] = pointer to start of the section of data in 'allReleventInfectedCitiesResult' for rank i
    int*                  largeCitiesByRank_length;             //[i] = number of big cities in rank i (length to read from head)


    //rank data
    int myRank;
    int numRanks;


    double runtime = 0;
    ticks computationTicks = 0;
    ticks passingTicks = 0;
    ticks startTick = 0;

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
    largeInfectedCitiesResultByRank_head = (struct InfectedCity**) malloc(numRanks * sizeof(struct InfectedCity*));

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
    //HEISEN ERROR AFTER THESE PRINT STATEMENTS

    //init memory for infected cities
    covid_allocateMem_InfectedCities_init(&cityData, &allReleventInfectedCities, &allReleventInfectedCitiesResult, numRelevantCities);

    printf("rank %d allocated infectedCities; setting up city data\n", myRank);

    //set up cityData now the number of large cities in each rank is known
    setupCityData(&cityData, largeCitiesByRank_head,
        allReleventInfectedCities, largeInfectedCitiesByRank_head,
        allReleventInfectedCitiesResult, largeInfectedCitiesResultByRank_head,
        numSmallCities, largeCitiesByRank_length, myRank, numRanks);

    MPI_Barrier(MPI_COMM_WORLD);

    //HEISNE ERROR BEROFE HERE
    //debug
    printLargeCitySample(largeCitiesByRank_head, largeInfectedCitiesByRank_head, myRank, numRanks);

    MPI_passLargeCities(largeCitiesByRank_head, mpi_cityData_type, largeCitiesByRank_length, myRank, numRanks);

    printf("large cities passed\n");

    MPI_Barrier(MPI_COMM_WORLD);



    //PATIENT ZERO
    // printf("rank: %d patient zero\n", myRank);
    // if(myRank == 0){
        // printf("rank: %d infecting\n", myRank);
        // largeInfectedCitiesByRank_head[myRank]->susceptibleCount -= 1;
        // largeInfectedCitiesByRank_head[myRank]->infectedCount += 1;
    // }
    for(i = 0; i<cityDataLength; i++){
        if(strcmp(cityData[i].cityName, "Brooklyn") == 0){
            printf("rank %d infecting city %s %s\n", myRank, cityData[i].cityName, cityData[i].state);
            allReleventInfectedCities[i].susceptibleCount -= 1;
            allReleventInfectedCities[i].infectedCount += 1;
        }
    }
    // printf("rank: %d patient zero infected\n", myRank);

    //debug
    printLargeCitySample(largeCitiesByRank_head, largeInfectedCitiesByRank_head, myRank, numRanks);
    printNYCities(cityData, allReleventInfectedCities, cityDataLength, myRank);

    MPI_Barrier(MPI_COMM_WORLD);

    //INFECTION ITERATIONS

    for(i = 0; i<iterations; i++){

        printf("rank %d, iteration %d\n", myRank, i);

        //intra-city update
        startTick = getticks();
        covid_intracity_kernelLaunch(&cityData,
            &allReleventInfectedCities,
            &allReleventInfectedCitiesResult,
            cityDataLength,
            threadsCount);
        computationTicks += (getticks() - startTick);
        data_result_swap(&allReleventInfectedCities, &allReleventInfectedCitiesResult,
            &largeInfectedCitiesByRank_head, &largeInfectedCitiesResultByRank_head);


        MPI_Barrier(MPI_COMM_WORLD);

        //debug
        printLargeCitySample(largeCitiesByRank_head, largeInfectedCitiesByRank_head, myRank, numRanks);
        MPI_Barrier(MPI_COMM_WORLD);
        printNYCities(cityData, allReleventInfectedCities, cityDataLength, myRank);
        MPI_Barrier(MPI_COMM_WORLD);

        //pass infectedCount of all cities to all other ranks
        startTick = getticks();
        MPI_passInfectionData(largeInfectedCitiesByRank_head, largeCitiesByRank_length, myRank, numRanks, mpi_infectedCity_type, i);
        passingTicks += (getticks() - startTick);

        //spread of desease
        // startTick = getticks();
        // covid_spread_kernelLaunch(&cityData,
        //     &allReleventInfectedCities,
        //     &allReleventInfectedCitiesResult,
        //     numSmallCities,
        //     numLargeCitiesWithinRank,
        //     allLargeCityCount,
        //     threadsCount);
        // computationTicks += (getticks() - startTick);
        // data_result_swap(&allReleventInfectedCities, &allReleventInfectedCitiesResult,
        //     &largeInfectedCitiesByRank_head, &largeInfectedCitiesResultByRank_head);

    }
    


    //write results



    //timing data
    if(myRank == 0){
        runtime = MPI_Wtime();
        printf("MPI running time: %f\n", runtime);
        printf("computationTicks: %llu\n", computationTicks);
        printf("passingTicks:     %llu\n", passingTicks);
    }


    //finalize
    MPI_Finalize();

    //free memory
    covid_freeMem(&cityData, &allReleventInfectedCities, &allReleventInfectedCitiesResult);
    free(largeCitiesByRank_length);
    free(largeCitiesByRank_head);
    free(largeInfectedCitiesByRank_head);
    free(largeInfectedCitiesResultByRank_head);


}

MPI_Datatype getCityDataDataType()
{

    const int fieldCount = 7;
    int blocklengths [fieldCount] = {50, 3, 1, 1, 1, 1, 1};
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

    MPI_File_open(MPI_COMM_WORLD, fileName, MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
    MPI_File_read_at(f, startPos, buf, bufSize, MPI_CHAR, &status);

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
                    strcpy(curState, token);
                    if(onFinalState && !finalStateRecorded){
                        strcpy(finalState, curState);
                        finalStateRecorded = 1;
                    }
                    if(onFinalState && strcmp(curState, finalState) != 0){
                        i = bufSize;
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
    for(i = smallCityIndex; i < smallCityIndex + largeCityIndex; i++){
        (*cityData)[i] = largeCities[i - smallCityIndex];
    }
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
        // printf("rank %d recieving %d from rank %d\n", myRank, largeCitiesByRank_length[i], i);
        // Exchange row data with MPI Ranks using MPI_Isend/Irecv.
        //                 data                               length             data type  source   tag        MPI_COMM    MPI_Request
        MPI_Irecv(largeInfectedCitiesByRank_head[i], largeCitiesByRank_length[i], MPI_INT,     i,    tag + i,  MPI_COMM_WORLD, &request0);
    }

    //send data
    for(i = 0; i<numRanks; i++){
        if(i==myRank) continue;
        // printf("rank %d sending %d to rank %d\n", myRank, largeCitiesByRank_length[myRank], i);
        //                 data                              length                   data type          dest     tag       MPI_COMM     MPI_Request
        MPI_Isend(largeInfectedCitiesByRank_head[i], largeCitiesByRank_length[i], mpi_infectedCity_type,   i,   tag + i,  MPI_COMM_WORLD, &request1);
    }

    MPI_Wait(&request0, &status);
    MPI_Wait(&request1, &status);
}

void MPI_passOutputDataLen(int** outputDataLens, int myRank, int numRanks){

    int i;

    MPI_Request request0, request1;
    MPI_Status status;

    for(i = 0; i<numRanks; i++){
        if(i==myRank) continue;
        // Exchange row data with MPI Ranks using MPI_Isend/Irecv.
        //                  data               length   data type  source  tag    MPI_COMM    MPI_Request
        MPI_Irecv(*outputDataLens+i,    1,     MPI_INT,     i,    1,  MPI_COMM_WORLD, &request0);
    }

    //send the number of large cities in my rank to all other ranks
    for(i = 0; i<numRanks; i++){
        if(i==myRank) continue;
        //                  data                    length  data type    dest  tag     MPI_COMM     MPI_Request
        MPI_Isend(*outputDataLens+myRank,    1,    MPI_INT,     i,    1,  MPI_COMM_WORLD, &request1);
    }

    MPI_Wait(&request0, &status);
    MPI_Wait(&request1, &status);

}

//takes in list of fileNames, cityData, infectedCityData
//and outputs cityName, state abbrev, city ranking, total population, susceptibleCount, infectedCount
//recoveredCount, deceasedCount in csv format
/*double outputData(int numFiles, char*** fileNames, int rank, int numRanks, struct City** cityData, 
struct InfectedCity** infectedCityData, int numCurRankCities){

    int i, myFile;
    char* outputStr;
    char* headerStr;
    char tmpStr[300] = {'\0'};
    int* outputDataLens;
    MPI_File f;
    MPI_Offset myOffset=0;
    outputStr = (char*)calloc(numCurRankCities*150, sizeof(char));
    headerStr = (char*) calloc(500, sizeof(char));
    outputDataLens = (int*) calloc(numRanks, sizeof(int));

    strcat(headerStr, "iteration: " + itoa(infectedCityData[0].iterationOfInfection) + 
    "\ncityName, state abbrev, city ranking, total population, susceptibleCount, infectedCount
    recoveredCount, deceasedCount\n");
    
    myFile = numRanks % numFiles;
    if(rank == myFile){
        strcat(outputStr, headerStr);
    }

    for(i = 0; i < numCurRankCities; i++){
        strcat(outputStr, "\"" + (*cityData)[i].cityName + "\",");
        strcat(outputStr, "\"" + (*cityData)[i].state + "\",");
        strcat(outputStr, "\"" + itoa((*cityData)[i].cityRanking) + "\",");
        strcat(outputStr, "\"" + itoa((*cityData)[i].totalPopulation) + "\",");
        strcat(outputStr, "\"" + itoa((*infectedCityData)[i].susceptibleCount) + "\",");
        strcat(outputStr, "\"" + itoa((*infectedCityData)[i].infectedCount) + "\",");
        strcat(outputStr, "\"" + itoa((*infectedCityData)[i].recoveredCount) + "\",");
        strcat(outputStr, "\"" + itoa((*infectedCityData)[i].deceasedCount) + "\"\n");
    }
    outputDataLens[rank] = strlen(outputStr);
    MPI_passOutputDataLen(&outputDataLens, rank, numRanks);

        //start time
    t1 = MPI_Wtime();

    //open the right file
    MPI_File_open(MPI_COMM_WORLD, (*fileNames)[myFile], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f);

    //set view based on outputDataLen
    //loop through all ranks sharing the file that are aheard of us
    for(i = myFile; i < rank; i += numFiles)
        myOffset += outputDataLens[i];
    MPI_File_set_view(f, myOffset, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);
    MPI_File_write(f, outputStr, strlen(outputStr), MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&f);
    t2 = MPI_Wtime();

    return t2 - t1;
}*/