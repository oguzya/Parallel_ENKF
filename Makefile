CC = g++
CFLAGS = -g -Wall -O3 -fopenmp
LDLIBS = -I eigen -lstdc++
OBJECTS = main.o new_util.o

all: main

new_util.o: new_util.cpp new_util.h

main.o: main.cpp


main: $(OBJECTS) new_util.h
	$(CC) -o main $(CFLAGS) $(LDLIBS) $(OBJECTS)

clean:
	rm -f *.o *~ main
