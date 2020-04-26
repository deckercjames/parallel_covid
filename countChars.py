import sys

file = open('uscities.csv', 'r')

chars = 0
for line in file:
    chars += len(line)
print(chars)
file.close()