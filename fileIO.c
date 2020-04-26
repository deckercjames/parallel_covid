#include "fileIO.h"
#include <string.h>
#include <math.h>

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