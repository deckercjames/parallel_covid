#include "fileIO.h"
#include <string.h>
#include <math.h>
#include "covidTypes.h"

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
    
    printf("rank %d before reading in\n", rank);
    for(i = 0; i < numFiles; i++){
        
        if(startPos < charCounter + (*numCharsByFile)[i] && readEndPos > charCounter){//read a file
            if(charCounter + (*numCharsByFile)[i] < readEndPos){
                tmpBufSize = (*numCharsByFile)[i] - (startPos + curChar - charCounter);
            }else{
                tmpBufSize = readEndPos - startPos - curChar;
            }
            printf("before file open tmpBufSize: %d startPos: %d i: %d rank: %d curChar: %d charCounter: %d\n", 
            tmpBufSize, startPos, i, rank, curChar, charCounter);
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

    printf("rank %d before main loop\n", rank);
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
                        //printf("rank: %d, state: %s, smallI: %d, largeI: %d, i: %d\n", 
                        //rank, token, smallCityIndex, largeCityIndex, i);
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
    printf("rank %d before second loop\n", rank);
    for(i = smallCityIndex; i < smallCityIndex + largeCityIndex; i++){
        (*cityData)[i] = largeCities[i - smallCityIndex];
    }
    printf("rank %d after second loop\n", rank);
    *cityDataLength = smallCityIndex + largeCityIndex;
    *numSmallCities = smallCityIndex;
    free(buf);
    free(f);
    return t2 - t1;
}