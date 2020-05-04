typedef struct City City;
struct City{

    char cityName[50];
    char state[3];

    int cityRanking;
    
    int totalPopulation;
    int density;

    double lattitude;
    double longitude;

};

struct InfectedCity{
    int susceptibleCount;
    int infectedCount;
    int recoveredCount;
    int deceasedCount;
    int iterationOfInfection;
};