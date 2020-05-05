all: covid19-main.c covid19-with-cuda.cu
	mpicc -g covid19-main.c -c -o covid19-main.o
	nvcc -g -G -arch=sm_70 covid19-with-cuda.cu -c -o covid19-cuda.o
	mpicc -g covid19-main.o covid19-cuda.o -o covid-exe -L/usr/local/cuda-10.1/lib64/ -lcudadevrt -lcudart -lstdc++
