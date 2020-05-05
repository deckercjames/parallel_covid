
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<string.h>

#include "mpi.h"
#include "covidTypes.h"



typedef unsigned long long ticks;



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
double readFile(int numFiles, char*** fileName, int numChars, int** numCharsByFile, int rank, int numRanks, 
struct City** cityData, int* cityDataLength, int* numSmallCities);

//takes in list of fileNames, cityData, infectedCityData
//and outputs cityName, state abbrev, city ranking, total population, susceptibleCount, infectedCount
//recoveredCount, deceasedCount in csv format
double outputData(int numFiles, char*** fileNames, int rank, int numRanks, struct City** cityData, 
struct InfectedCity** infectedCityData, int numCurRankCities);

const char filename[50] = "WeakScalingFiles/01Nodes_443464.csv";
const int fileLength = 443464;

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

    if(numRanks == 1) return;

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

void MPI_passLargeCities(struct City* cityData,
                            struct City* cityDataPassing,
                            int* offsets, 
                            int* lengths, 
                            int numSmallCities,
                            int numAllLargeCities,
                            MPI_Datatype mpi_cityData_type,
                            int myRank, int numRanks)
{

    int i;

    MPI_Request request0, request1;
    MPI_Status status;

    if(numRanks == 1) return;

    printf("passing large cities\n");

    //copy device data to data that can be passed
    for(i = 0; i<numAllLargeCities; i++){
        cityDataPassing[ i ] = cityData[ numSmallCities + i ];
    }

    printf("copied data to passing array\n");

    //receive the large cites from all other ranks
    for(i = 0; i<numRanks; i++){
        if(i==myRank) continue;
        // Exchange row data with MPI Ranks using MPI_Isend/Irecv.
        //                  data                length         data type       source  tag    MPI_COMM     MPI_Request
        MPI_Irecv(cityDataPassing + offsets[i], lengths[i], mpi_cityData_type,    i,    1,  MPI_COMM_WORLD, &request0);
    }

    //send the large cities in my rank to all other ranks
    for(i = 0; i<numRanks; i++){
        if(i==myRank) continue;
        //                  data                        length          data type       dest   tag     MPI_COMM     MPI_Request
        MPI_Isend(cityDataPassing + offsets[myRank], lengths[myRank], mpi_cityData_type,  i,    1,  MPI_COMM_WORLD, &request1);
    }

    MPI_Wait(&request0, &status);
    MPI_Wait(&request1, &status);

    printf("mpi passing complete\n");

    //copy back to device
    for(i = 0; i<numAllLargeCities; i++){
        cityData[ numSmallCities + i ] = cityDataPassing[ i ];
    }

    printf("large cities copied back\n");

}

/*
Uses the 'numSmallCities' and numLargeCities for each rank to reallocate the correct amount of memory
for cityData and copy all existing cities to the new array. Also calculated the head of each rank's data
*/
void setupCityData(struct City* cityData,
                        int* offsets,
                        int* lengths,
                        int numCitiesWithinRank,
                        int numCities,
                        int myRank, int numRanks)
{

    int i;

    struct City* cityDataTemp;

    //this is to calculate the headPointers
    int offset = 0;

    //Increase the length of the cityData to hold all Large Cities from other ranks
    // cityData = (struct City*) realloc(cityData, totalLength * sizeof(struct City));
    covid_reallocateMem_CityData(&cityData, &cityDataTemp, numCitiesWithinRank, numCities);

    //calculate the head of each rank's large cities

    printf("rank %d, reallocation complete\n", myRank);

    //first it the large cities in this rank
    offsets[myRank] = offset;
    printf("rank %d:  offset: %d \n", myRank, offsets[myRank]);
    offset += lengths[myRank];
    printf("rank %d:  length: %d \n", myRank, lengths[myRank]);
    //next, all other rank's large cities
    for(i = 0; i<numRanks; i++){
        if(i == myRank) continue;//ignore my rank

        printf("rank %d: i: %d offset: %d length[%d]: %d\n", myRank, i, offset, i, lengths[i]);

        //set the head of the large city section for rank i
        offsets[i] = offset;
        offset += lengths[i];
    }

    printf("rank %d: ending offset: %d\n", myRank, offset);

}


void MPI_passInfectionData(struct InfectedCity* infectedCityData,
                            struct InfectedCity* infectedCityDataPassing,
                            int* offsets, 
                            int* lengths, 
                            int numSmallCities,
                            int numAllLargeCities,
                            MPI_Datatype mpi_infectedCity_type,
                            int myRank, int numRanks);



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

        printf("\n");
    }

}


int main(int argc, char *argv[])
{

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
    int* largeCitiesByRank_offset; //[i] = begening of large city section for rank i (from begenning of all large cities); offsets[myRank] = 0
    int* largeCitiesByRank_length; //[i] = number of big cities in rank i (length to read from head)

    //temp data for passing between ranks
    struct City* cityDataPassing;
    struct InfectedCity* infectedCityDataPassing;

    //rank data
    int myRank;
    int numRanks;


    double runtime = 0;
    ticks computationTicks = 0;
    ticks passingTicks = 0;
    ticks startTick = 0;


    //commandline arguemtns
    int iterations;
    int threadsCount;
    if(argc != 3){
        printf("This Covid-19 Simulator takes two arguemtns: The number of iterations and the number of threads per block\n");
    }
    iterations = atoi(argv[1]);
    threadsCount = atoi(argv[2]);


    // Setup MPI
    MPI_Init(&argc, &argv);

    //get the data types
    MPI_Datatype mpi_infectedCity_type = getInfectedCityDataType();
    MPI_Datatype mpi_cityData_type = getCityDataDataType();

    printf("test 2\n");

    //get my rank
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);



    char** outputFileNames;
    char** fileNames = (char**) calloc(1, sizeof(char*));
    fileNames[i] = (char*) calloc(30, sizeof(char));
    int* numCharsByFile = (int*)calloc(1, sizeof(int));
    //read in city Data
    fileNames[0] = "uscities.csv";
    numCharsByFile[0] = 5309769;
    readFile(1, &fileNames, fileLength, &numCharsByFile, myRank, numRanks, &cityData, &cityDataLength, &numSmallCities);

    //set the number of large cities within this rank
    numLargeCitiesWithinRank = cityDataLength - numSmallCities;

    //PASS LARGE CITY DATA TO ALL OTHER RANKS

    largeCitiesByRank_length = (int*) malloc(numRanks * sizeof(int));
    largeCitiesByRank_offset = (int*) malloc(numRanks * sizeof(int));

    //first pass the number of large cities to all other ranks so they know how many to recieve
    largeCitiesByRank_length[myRank] = numLargeCitiesWithinRank;
    MPI_passNumberOfLargeCities(largeCitiesByRank_length, myRank, numRanks);

    MPI_Barrier(MPI_COMM_WORLD);

    //calculate the total number of large cities
    allLargeCityCount = 0;
    for(i = 0; i<numRanks; i++){
        printf("rank %d: adding [%d] %d\n", myRank, 0, largeCitiesByRank_length[i]);
        allLargeCityCount += largeCitiesByRank_length[i];
    }
    printf("rank %d: allLArgeCityCount: %d\n", myRank, allLargeCityCount);
    numRelevantCities = numSmallCities + allLargeCityCount;

    //allocate memory for passing
    cityDataPassing = (struct City*) malloc(allLargeCityCount * sizeof(struct City));
    infectedCityDataPassing = (struct InfectedCity*) malloc(allLargeCityCount * sizeof(struct InfectedCity));

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
    setupCityData(cityData,
        largeCitiesByRank_offset,
        largeCitiesByRank_length,
        cityDataLength,
        numRelevantCities,
        myRank, numRanks);
    printf("rank %d: numLargeCities: %d \n", myRank, allLargeCityCount);

    MPI_Barrier(MPI_COMM_WORLD);

    //HEISNE ERROR BEROFE HERE
    //debug
    // printLargeCitySample(largeCitiesByRank_head, largeInfectedCitiesByRank_head, myRank, numRanks);
    printf("city data setup\n");

    MPI_passLargeCities(cityData,
                            cityDataPassing,
                            largeCitiesByRank_offset, 
                            largeCitiesByRank_length, 
                            numSmallCities,
                            allLargeCityCount,
                            mpi_cityData_type,
                            myRank, numRanks);

    printf("large cities passed\n");

    MPI_Barrier(MPI_COMM_WORLD);



    //PATIENT ZERO
    for(i = 0; i<cityDataLength; i++){
        if(strcmp(cityData[i].cityName, "Chain Lake") == 0 && strcmp(cityData[i].state, "WA") == 0){
            printf("rank %d infecting city %s %s ===============================\n", myRank, cityData[i].cityName, cityData[i].state);
            allReleventInfectedCities[i].susceptibleCount -= 1;
            allReleventInfectedCities[i].infectedCount += 1;
        }
    }

    //debug
    // printLargeCitySample(largeCitiesByRank_head, largeInfectedCitiesByRank_head, myRank, numRanks);
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


        MPI_Barrier(MPI_COMM_WORLD);

        //pass infectedCount of all cities to all other ranks
        startTick = getticks();
        MPI_passInfectionData(allReleventInfectedCities,
                            infectedCityDataPassing,
                            largeCitiesByRank_offset, 
                            largeCitiesByRank_length, 
                            numSmallCities,
                            allLargeCityCount,
                            mpi_infectedCity_type,
                            myRank, numRanks);
        passingTicks += (getticks() - startTick);

        //spread of desease
        startTick = getticks();
        covid_spread_kernelLaunch(&cityData,
             &allReleventInfectedCities,
             &allReleventInfectedCitiesResult,
             numSmallCities,
             numLargeCitiesWithinRank,
             allLargeCityCount,
             threadsCount);
            computationTicks += (getticks() - startTick);

    }
    
    outputFileNames = (char**) calloc(12, sizeof(char*));
    outputFileNames[0] = "covidOutput0.txt";
    outputFileNames[1] = "covidOutput1.txt";
    //write results
    outputData(2, &outputFileNames, myRank, numRanks, &cityData, 
    &allReleventInfectedCitiesResult, numSmallCities + numLargeCitiesWithinRank);


    MPI_Barrier(MPI_COMM_WORLD);
    printNYCities(cityData, allReleventInfectedCities, cityDataLength, myRank);
    MPI_Barrier(MPI_COMM_WORLD);


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

    free(largeCitiesByRank_offset);
    free(largeCitiesByRank_length);

    free(cityDataPassing);
    free(infectedCityDataPassing);


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


void MPI_passInfectionData(struct InfectedCity* infectedCityData,
                            struct InfectedCity* infectedCityDataPassing,
                            int* offsets, 
                            int* lengths, 
                            int numSmallCities,
                            int numAllLargeCities,
                            MPI_Datatype mpi_infectedCity_type,
                            int myRank, int numRanks)
{

    int i;

    MPI_Request request0, request1;
    MPI_Status status;

    if(numRanks == 1) return;

    //copy device data to data that can be passed
    for(i = 0; i<numAllLargeCities; i++){
        infectedCityDataPassing[ i ] = infectedCityData[ numSmallCities + i ];
    }

    //receive the large cites from all other ranks
    for(i = 0; i<numRanks; i++){
        if(i==myRank) continue;
        // Exchange row data with MPI Ranks using MPI_Isend/Irecv.
        //                  data                        length         data type           source  tag    MPI_COMM     MPI_Request
        MPI_Irecv(infectedCityDataPassing + offsets[i], lengths[i], mpi_infectedCity_type,    i,    1,  MPI_COMM_WORLD, &request0);
    }

    //send the large cities in my rank to all other ranks
    for(i = 0; i<numRanks; i++){
        if(i==myRank) continue;
        //                  data                                  length            data type       dest   tag     MPI_COMM     MPI_Request
        MPI_Isend(infectedCityDataPassing + offsets[myRank], lengths[myRank], mpi_infectedCity_type,  i,    1,  MPI_COMM_WORLD, &request1);
    }

    MPI_Wait(&request0, &status);
    MPI_Wait(&request1, &status);

    //copy back to device
    for(i = 0; i<numAllLargeCities; i++){
        infectedCityData[ numSmallCities + i ] = infectedCityDataPassing[ i ];
    }

}




double readFile(int numFiles, char*** fileName, int numChars, int** numCharsByFile, int rank, int numRanks, 
struct City** cityData, int* cityDataLength, int* numSmallCities){
    MPI_File* f;
    MPI_Status status;
    char* buf;
    char* endptr;
    int startPos = numChars * ((float) rank / numRanks);
    int endPos = numChars * (((float) rank + 1)/ numRanks);
    int bufSize = 2*numChars/numRanks;
    int tmpBufSize;
    int i, tokenNum, smallCityIndex = -1, largeCityIndex = -1, curChar = 0, charCounter = 0;
    char token[3000] = {'\0'};
    char tmpChar = '\0';
    static const struct City blankCity;
    struct City tmpCity;
    struct City largeCities[4000];
    char firstState[5] = {'\0'};
    char curState[5] = {'\0'};
    char finalState[5] = {'\0'};
    short firstStateRecorded = 0, onFinalState = 0, finalStateRecorded = 0;
    double t1, t2;
    if(rank == numRanks - 1){
        //ceiling of numChars/numRanks
        bufSize = numChars/numRanks + (numChars % numRanks != 0);
    }
    int readEndPos = startPos + bufSize;

    //we will be reading twice the expected size at each rank
    //because most ranks will have to go over expected size
    buf = (char*) calloc(bufSize, sizeof(char));
    *cityData = (struct City*) calloc(bufSize/120, sizeof(struct City));//approx 180 chars per line
    f = (MPI_File*) calloc(numFiles, sizeof(MPI_File));

    //start time
    t1 = MPI_Wtime();

    //open all the files
    for(i = 0; i < numFiles; i++){
        MPI_File_open(MPI_COMM_WORLD, (*fileName)[i], MPI_MODE_RDONLY, MPI_INFO_NULL, &f[i]);
    }

    //read into buf
    //curChar refers to local rank, charCounter refers to the num of chars in
    //all the files that have been passed
    
    for(i = 0; i < numFiles; i++){
        
        if(startPos < charCounter + (*numCharsByFile)[i] && readEndPos > charCounter){//read a file
            if(charCounter + (*numCharsByFile)[i] < readEndPos){
                tmpBufSize = (*numCharsByFile)[i] - (startPos + curChar - charCounter);
            }else{
                tmpBufSize = readEndPos - startPos - curChar;
            }
            //printf("before read at\n");
            MPI_File_read_at(f[i], startPos + curChar - charCounter, &buf[curChar], tmpBufSize, MPI_CHAR, &status);
            curChar += tmpBufSize;
        }
        charCounter += (*numCharsByFile)[i];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for(i = 0; i < numFiles; i++){
        MPI_File_close(&f[i]);
    }
    t2 = MPI_Wtime();

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
                        //printf("final state: %s cur state: %s\n", finalState, curState);
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
                    if(rank != 0 && strcmp(curState, firstState) == 0)
                        break;
                    if(tmpCity.cityRanking < 3){
                        largeCities[largeCityIndex] = tmpCity;
                        largeCityIndex++;
                    }else{
                        if(tmpCity.totalPopulation < 10000)
                            tmpCity.cityRanking = 4;
                        if(tmpCity.totalPopulation < 5000)
                            tmpCity.cityRanking = 5;
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
    free(f);
    return t2 - t1;
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
double outputData(int numFiles, char*** fileNames, int rank, int numRanks, struct City** cityData, 
struct InfectedCity** infectedCityData, int numCurRankCities){

    int i, myFile;
    char* outputStr;
    char* headerStr;
    char tmpStr[300] = {'\0'};
    int* outputDataLens;
    double t1, t2;
    char miniTmpStr[100] = {'\0'};
    MPI_File* f;
    MPI_Offset myOffset=0;
    outputStr = (char*)calloc(numCurRankCities*150, sizeof(char));
    headerStr = (char*) calloc(500, sizeof(char));
    outputDataLens = (int*) calloc(numRanks, sizeof(int));
    f = (MPI_File*) calloc(numFiles, sizeof(MPI_File));

    sprintf(miniTmpStr, "%d", infectedCityData[0]->iterationOfInfection); 
    strcat(headerStr, "iteration: ");
    strcat(headerStr, miniTmpStr);
    strcat(headerStr, "\ncityName, state abbrev, city ranking, total population, susceptibleCount, infectedCount, recoveredCount, deceasedCount\n");
    
    myFile = rank % numFiles;
    if(rank == myFile){
        strcat(outputStr, headerStr);
    }

    for(i = 0; i < numCurRankCities; i++){
        strcat(outputStr, "\""); strcat(outputStr, (*cityData)[i].cityName);
        strcat(outputStr, "\",\""); strcat(outputStr, (*cityData)[i].state);
        sprintf(miniTmpStr, "%d", (*cityData)[i].cityRanking); 
        strcat(outputStr, "\",\""); strcat(outputStr,  miniTmpStr);
        sprintf(miniTmpStr, "%d", (*cityData)[i].totalPopulation); 
        strcat(outputStr, "\",\""); strcat(outputStr,  miniTmpStr);
        sprintf(miniTmpStr, "%d", (*cityData)[i].cityRanking); 
        strcat(outputStr, "\",\""); strcat(outputStr,  miniTmpStr);
        sprintf(miniTmpStr, "%d", (*infectedCityData)[i].susceptibleCount); 
        strcat(outputStr, "\",\""); strcat(outputStr, miniTmpStr);
        sprintf(miniTmpStr, "%d", (*infectedCityData)[i].infectedCount); 
        strcat(outputStr, "\",\""); strcat(outputStr, miniTmpStr);
        sprintf(miniTmpStr, "%d", (*infectedCityData)[i].recoveredCount); 
        strcat(outputStr, "\",\""); strcat(outputStr, miniTmpStr);
        sprintf(miniTmpStr, "%d", (*infectedCityData)[i].deceasedCount);
        strcat(outputStr, "\",\""); strcat(outputStr, miniTmpStr);
        strcat(outputStr, "\"\n");
    }
    outputDataLens[rank] = strlen(outputStr);
    MPI_passOutputDataLen(&outputDataLens, rank, numRanks);

        //start time
    t1 = MPI_Wtime();
    printf("rank %d myFile: %d len[0]: %d len[1]: %d len[2]: %d len[3]: %d filename: %s output: %c\n",
    rank, myFile, outputDataLens[0], outputDataLens[1], outputDataLens[2], outputDataLens[3],
    (*fileNames)[myFile], outputStr[1]);

    //open all the files
    for(i = 0; i < numFiles; i++){
        MPI_File_open(MPI_COMM_WORLD, (*fileNames)[i], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f[i]);
    }

    //set view based on outputDataLen
    //loop through all ranks sharing the file that are aheard of us
    for(i = myFile; i < rank; i += numFiles)
        myOffset += outputDataLens[i];
    for(i = 0; i < numFiles; i++)
        MPI_File_set_view(f[i], myOffset, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);
    MPI_File_write(f[myFile], outputStr, strlen(outputStr), MPI_CHAR, MPI_STATUS_IGNORE);
    //MPI_File_write_at(f[myFile], myOffset, outputStr, strlen(outputStr), MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);
    //close all the files
    for(i = 0; i < numFiles; i++)
        MPI_File_close(&f[i]);
    printf("rank %d done\n", rank);
    t2 = MPI_Wtime();
    free(outputStr);
    free(headerStr);
    free(outputDataLens);
    return t2 - t1;
}