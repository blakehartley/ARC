# Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This Makefile was modified for use in the compiling of ARC by Blake Teixeira Hartley.

# bluecrab variables (not sure what is best here):
MPI_HOME = /software/apps/mpi/openmpi/3.1.3a1

CUDA_PATH = /software/apps/cuda/9.2/
CUDA_INCDIR = /software/apps/cuda/9.2/include
CUDA_LIB64DIR = /software/apps/cuda/9.2/lib64

# Compilers
MPICC=$(PREP) mpicc
MPILD=$(PREP) mpic++
NVCC=$(PREP) nvcc

# Flags
CFLAGS = -std=c99 -O3 -march=native -Wall
CFLAGS += -I$(CUDA_INCDIR)
#CFLAGS += -L/etc/glue/nvidia/lib64 -Wl,-rpath,/etc/glue/nvidia/lib64 -lcudart

LDFLAGS = -L$(CUDA_LIB64DIR) -Wl,-rpath,$(CUDA_LIB64DIR) -lcudart
#LDFLAGS += -L/etc/glue/nvidia/lib64 -Wl,-rpath,/etc/glue/nvidia/lib64

MPICFLAGS=-I${MPI_HOME}/include
CUDACFLAGS=-I${CUDA_INCDIR}

#CUDACFLAGS=-I${CUDA_INSTALL_PATH}/include

GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM35    := -gencode arch=compute_37,code=sm_37
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
#GENCODE_SM52    := -gencode arch=compute_52,code=\"sm_52,compute_52\"
GENCODE_FLAGS   := $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM37) $(GENCODE_SM50)
#$(GENCODE_SM52)

NVCCFLAGS=-O3 $(GENCODE_FLAGS) -Xcompiler -march=native

CUDALDFLAGS := -L${CUDA_INCDIR}/lib64 -lcudart

# Description of binaries
BINDIR=./bin
ARC=$(BINDIR)/arc
BINARIES=$(ARC)

# Commands
all: $(BINARIES)

test:
	@echo $(MPI_HOME)
	@echo $(CUDA_INCDIR)
	@echo $(CUDA_LIB64DIR)
	@echo $(CFLAGS)
	@echo $(LDFLAGS)

dev.o: ./src/arc.h ./src/arc_dev.cu Makefile
	$(NVCC) $(MPICFLAGS) $(NVCCFLAGS) -c ./src/arc_dev.cu -o ./src/dev.o

host.o: ./src/arc.h ./src/arc_host.cpp Makefile
	$(MPILD) $(LDFLAGS) $(CFLAGS) -c ./src/arc_host.cpp -o ./src/host.o
	
$(ARC): dev.o host.o Makefile
	mkdir -p $(BINDIR)
	$(MPILD) $(LDFLAGS) -o $(ARC) ./src/dev.o ./src/host.o 
	
clean:
	rm -rf ./src/*.o ./src/*~ ./src/inc/*~ $(BINARIES)

cleandat:
	rm -rf ./dat/hydrogen/* ./dat/helium/* ./dat/photons/* ./dat/background/*

makedir:
	mkdir ./dat/ ./dat/hydrogen ./dat/helium ./dat/photons/ ./dat/background/ ./bin/
