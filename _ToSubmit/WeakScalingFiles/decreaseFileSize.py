import sys

# file = open('uscities.csv', 'r')

with open('Full.csv') as fp:
	line = fp.readline()
	cnt = 0
	while line:
		if cnt % 3 == 0 or line.startswith('"Chain Lake","Chain Lake","WA"'):
			print line,
		line = fp.readline()
		cnt += 1

# file.close('Full.csv')