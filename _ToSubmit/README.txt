This program will run with the full city data set. There are no arguments. We changed parameters by changing the c file. It will run for 60 iteration with 64 threads per block.

To Test Weak Scaling:
- The const field filename (line 74) should be changed to the filename of a reduced file
- The const field fileLength (line 75) should be changed to the number of characters in the file (included in the filename)
- ex:
const char filename[50] = "WeakScalingFiles/06Nodex_2668414.cvs";
const int fileLength = 2668414;