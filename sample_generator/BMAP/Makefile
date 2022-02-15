#=========================================================================#
#                 BMAP-Sim : Simulate a multi-tiered system               #
#                     by A. Riska and E. Smirni                           #
#               (riska@cs.wm.edu    esmirni@cs.wm.edu)                    #
#=========================================================================#

CFLAGS = -g -Wall

CC  = g++

#SRC = BMAP-Trace.cc BMAP.cc rngs.cc rvgs.cc
#OBJ = $(SRC:.cc=.o)

#OBJ:$(SRC)
#	$(CC) $(CFLAGS) -c $^

OBJ = BMAP-Trace.o BMAP.o rngs.o rvgs.o
.cc.o:
	$(CC) $(CFLAGS) -c $<

all: BMAP-Trace

.PHONY : all

BMAP-Trace:$(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o BMAP-Trace -lm

clean:
	rm -f *.o
	rm -f *~

