import sys

file = open('04Nodes_.csv', 'r')

chars = 0
for line in file:
    chars += len(line)
print(chars)
file.close()