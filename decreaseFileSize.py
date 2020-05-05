import sys

# file = open('uscities.csv', 'r')

with open('Full.csv') as fp:
	line = fp.readline()
	cnt = 0
	while line:
		if cnt % 2 == 0:
			print(line.strip)
			line = fp.readline()
			cnt += 1

file.close()