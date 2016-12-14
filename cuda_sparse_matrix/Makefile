CXX := g++
CC := gcc 
LINK := nvcc
NVCC := nvcc -ccbin /usr/bin
CFLAGS := -g

INCLUDES    = -I. $$CUDA_INC
LIBS        = $$CUDA_LIB

OBJS = mmio.c.o main.c.o kernels.cu.o pagerank.c.o poisson.c.o
TARGET = sparse
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIBS)

.SUFFIXES: .c .cpp .cu .o

$(TARGET): $(OBJS)
	$(LINKLINE)
%.c.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
%.cu.o: %.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) -c $< -o $@
clean:
	rm -rf *.o $(TARGET)
all: $(TARGET)

