#include "fileIO.h"
#include<stdio.h>
#include<stdlib.h>
#include "covidTypes.h"

int main(int argc, char *argv[]){
    int rank, numRanks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int cityDataLength, numSmallCities, i;
    struct City* cityData1;
    struct City* cityData2;
    struct City* cityData4;
    struct City* cityData6;
    struct City* cityData12;
    const char filename[20] = "uscities.csv";
    char** fileNames = (char**) calloc(12, sizeof(char*));
    double  time;
    for(i = 0; i < 12; i++){
        fileNames[i] = (char*) calloc(30, sizeof(char));
    }
    int* numCharsByFile = (int*)calloc(12, sizeof(int));

    fileNames[0] = "uscities.csv";
    numCharsByFile[0] = 5309769;
    time = readFile(1, &fileNames, 5309769, &numCharsByFile, rank, numRanks, &cityData1, &cityDataLength, &numSmallCities);
    printf("1 file rank %d I/O time: %f\n", rank, time);

    fileNames[0] = "twoFiles/uscities0.txt";
    fileNames[1] = "twoFiles/uscities1.txt";
    numCharsByFile[0] = 2696995;
    numCharsByFile[1] = 2612774;
    time = readFile(2, &fileNames, 5309769, &numCharsByFile, rank, numRanks, &cityData2, &cityDataLength, &numSmallCities);
    printf("2 files rank %d I/O time: %f\n", rank, time);

    fileNames[0] = "fourFiles/uscities0.txt";
    fileNames[1] = "fourFiles/uscities1.txt";
    fileNames[2] = "fourFiles/uscities2.txt";
    fileNames[3] = "fourFiles/uscities3.txt";
    numCharsByFile[0] = 1339184;
    numCharsByFile[1] = 1357811;
    numCharsByFile[2] = 1309432;
    numCharsByFile[3] = 1303342;
    time = readFile(4, &fileNames, 5309769, &numCharsByFile, rank, numRanks, &cityData4, &cityDataLength, &numSmallCities);
    printf("4 files rank %d I/O time: %f\n", rank, time);

    fileNames[0] = "sixFiles/uscities0.txt";
    fileNames[1] = "sixFiles/uscities1.txt";
    fileNames[2] = "sixFiles/uscities2.txt";
    fileNames[3] = "sixFiles/uscities3.txt";
    fileNames[4] = "sixFiles/uscities4.txt";
    fileNames[5] = "sixFiles/uscities5.txt";
    numCharsByFile[0] = 893558;
    numCharsByFile[1] = 885053;
    numCharsByFile[2] = 918752;
    numCharsByFile[3] = 868535;
    numCharsByFile[4] = 872112;
    numCharsByFile[5] = 871759;
    //printf("before file read call\n");
    //readFile(filename, 5309769, rank, numRanks, &cityData, &cityDataLength, &numSmallCities);
    time = readFile(6, &fileNames, 5309769, &numCharsByFile, rank, numRanks, &cityData6, &cityDataLength, &numSmallCities);
    //printf("after file read call\n");
    printf("6 files rank %d I/O time: %f\n", rank, time);

    fileNames[0] = "twelveFiles/uscities0.txt";
    fileNames[1] = "twelveFiles/uscities1.txt";
    fileNames[2] = "twelveFiles/uscities2.txt";
    fileNames[3] = "twelveFiles/uscities3.txt";
    fileNames[4] = "twelveFiles/uscities4.txt";
    fileNames[5] = "twelveFiles/uscities5.txt";
    fileNames[6] = "twelveFiles/uscities6.txt";
    fileNames[7] = "twelveFiles/uscities7.txt";
    fileNames[8] = "twelveFiles/uscities8.txt";
    fileNames[9] = "twelveFiles/uscities9.txt";
    fileNames[10] = "twelveFiles/uscities10.txt";
    fileNames[11] = "twelveFiles/uscities11.txt";
    numCharsByFile[0] = 448936;
    numCharsByFile[1] = 444622;
    numCharsByFile[2] = 445807;
    numCharsByFile[3] = 439246;
    numCharsByFile[4] = 455647;
    numCharsByFile[5] = 463105;
    numCharsByFile[6] = 432625;
    numCharsByFile[7] = 435910;
    numCharsByFile[8] = 441059;
    numCharsByFile[9] = 431053;
    numCharsByFile[10] = 429542;
    numCharsByFile[11] = 442217;
    time = readFile(12, &fileNames, 5309769, &numCharsByFile, rank, numRanks, &cityData12, &cityDataLength, &numSmallCities);
    printf("12 files rank %d I/O time: %f\n", rank, time);



    //printf("cityDataLen: %d numSmallCities: %d rank: %d\n", cityDataLength, numSmallCities, rank);
    
    /*if(rank == 0){
        for(i = 0; i < cityDataLength; i += 100){
            printf("rank %d city %d population %d density %d lat %lf long %lf name %s state %s cityRanking %d\n", 
            rank, i, cityData[i].totalPopulation, cityData[i].density, cityData[i].lattitude,
            cityData[i].longitude, cityData[i].cityName, cityData[i].state, cityData[i].cityRanking);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 1){
        for(i = 0; i < cityDataLength; i += 100){
            printf("rank %d city %d population %d density %d lat %lf long %lf name %s state %s cityRanking %d\n", 
            rank, i, cityData[i].totalPopulation, cityData[i].density, cityData[i].lattitude,
            cityData[i].longitude, cityData[i].cityName, cityData[i].state, cityData[i].cityRanking);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);    
    if(rank == 2){
        for(i = 0; i < cityDataLength; i += 100){
            printf("rank %d city %d population %d density %d lat %lf long %lf name %s state %s cityRanking %d\n", 
            rank, i, cityData[i].totalPopulation, cityData[i].density, cityData[i].lattitude,
            cityData[i].longitude, cityData[i].cityName, cityData[i].state, cityData[i].cityRanking);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);    
    if(rank == 3){
        for(i = 0; i < cityDataLength; i += 100){
            printf("rank %d city %d population %d density %d lat %lf long %lf name %s state %s cityRanking %d\n", 
            rank, i, cityData[i].totalPopulation, cityData[i].density, cityData[i].lattitude,
            cityData[i].longitude, cityData[i].cityName, cityData[i].state, cityData[i].cityRanking);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);    
    if(rank == 4){
        for(i = 0; i < cityDataLength; i += 100){
            printf("rank %d city %d population %d density %d lat %lf long %lf name %s state %s cityRanking %d\n", 
            rank, i, cityData[i].totalPopulation, cityData[i].density, cityData[i].lattitude,
            cityData[i].longitude, cityData[i].cityName, cityData[i].state, cityData[i].cityRanking);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 5){
        for(i = 0; i < cityDataLength; i += 100){
            printf("rank %d city %d population %d density %d lat %lf long %lf name %s state %s cityRanking %d\n", 
            rank, i, cityData[i].totalPopulation, cityData[i].density, cityData[i].lattitude,
            cityData[i].longitude, cityData[i].cityName, cityData[i].state, cityData[i].cityRanking);
        }
    }
    for(i = 0; i < 12; i++){
        free(fileNames[i]);
    }
    free(fileNames);
    free(numCharsByFile);*/
    free(cityData1);
    free(cityData2);
    free(cityData4);
    free(cityData6);
    free(cityData12);
    MPI_Finalize();
}