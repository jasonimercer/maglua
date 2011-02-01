ARNAME=magnetostatics.a 

OBJECTS=cubature.o gamma_ab_v.o

CC=gcc
CFLAGS=-ansi -pedantic -O2 -g

all: ${ARNAME}

${ARNAME} : ${OBJECTS}
	ar cqs ${ARNAME} ${OBJECTS}
	
%.o : %.c %.h
	@ ${CC} $(CFLAGS) -c $*.c -o $*.o
	@ echo "CC $*"

clean:
	@ rm -f *~
	@ rm -f *.o
	@ echo "Temporary files removed"

distclean: clean
	@ rm -f ${ARNAME}
	@ echo "Library deleted"
