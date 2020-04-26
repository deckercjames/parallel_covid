#include "fileIO.h"
#include<stdio.h>
#include<stdlib.h>

int main(int argc, char *argv[]){
    int cityDataLength, numSmallCities, rank, numRanks, i;
    struct City* cityData;
    const char filename[20] = "uscities.csv";

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    printf("before file read call\n");
    readFile(filename, 5309769, rank, numRanks, &cityData, &cityDataLength, &numSmallCities);
    printf("after file read call\n");
    printf("cityDataLen: %d numSmallCities: %d rank: %d\n", cityDataLength, numSmallCities, rank);
    
    if(rank == 0){
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
    
    MPI_Finalize();
}