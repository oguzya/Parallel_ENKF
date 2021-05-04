CC = g++
CFLAGS = -g -Wall -O3 -fopenmp
LDLIBS = -I eigen -lstdc++
OBJECTS = main.o utilities.o

all: main

utilities.o: utilities.cpp utilities.h

main.o: main.cpp


main: $(OBJECTS) utilities.h
	$(CC) -o main $(CFLAGS) $(LDLIBS) $(OBJECTS)

clean:
	rm -f *.o *~ main
