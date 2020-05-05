This program will run with the full city data set. There are no arguments. We changed parameters by changing the c file. It will run for 60 iteration with 64 threads per block.

If you wish to run the program with multiple file input, specify the filenames and filelenghts (in chars) in the passed arrays.
ex.

    char** fileNames = (char**) calloc(6, sizeof(char*));
    for(i = 0; i < 6; i++){
        fileNames[i] = (char*) calloc(30, sizeof(char));
    }

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

   readFile(6, &fileNames, fileLength, &numCharsByFile, myRank, numRanks, &cityData, &cityDataLength, &numSmallCities);

We have included a folder with the input split into 6 files that can be used with the example code. A similar process can be used to run multiple file output with the outputData function.
This program will run with the full city data set. There are no arguments. We changed parameters by changing the c file. It will run for 60 iteration with 64 threads per block.

To Test Weak Scaling:
- The const field filename (line 74) should be changed to the filename of a reduced file
- The const field fileLength (line 75) should be changed to the number of characters in the file (included in the filename)
- ex:
const char filename[50] = "WeakScalingFiles/06Nodex_2668414.cvs";
const int fileLength = 2668414;
