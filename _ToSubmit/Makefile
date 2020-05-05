all: covid-main.c covid-with-cuda.cu
	mpicc -g covid-main.c -c -o covid-main.o
	nvcc -g -G -arch=sm_70 covid-with-cuda.cu -c -o covid-cuda.o
	mpicc -g covid-main.o covid-cuda.o -o covid-exe -L/usr/local/cuda-10.1/lib64/ -lcudadevrt -lcudart -lstdc++
