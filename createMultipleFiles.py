import sys

inputFile = open('uscities.csv', 'r')

curFileCount = 0
curFileNum = 0
outputFile = open("uscities" + str(curFileNum) + ".txt", "w+")
for line in inputFile:
    outputFile.write(line)
    curFileCount += 1
    if curFileCount > 2407 and curFileNum != 11:
        curFileNum += 1
        curFileCount = 0
        outputFile.close()
        outputFile = open("uscities" + str(curFileNum) + ".txt", "w+")
outputFile.close()