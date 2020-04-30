typedef struct City City;
struct City{
    int totalPopulation;
    int density;

    int cityRanking;
    double lattitude;
    double longitude;
    char cityName[50];
    char state[2];

    int connectedCities;
    struct City* connectedCitiesIndicies;
    double* edgeWeights;
};

struct InfectedCity{
    int susceptibleCount;
    int infectedCount;
    int recoveredCount;
    int deceasedCount;
    int iterationOfInfection;
};