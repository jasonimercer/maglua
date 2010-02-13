OBJECTS=main.o spinsystem.o spinoperation.o spinoperationexchange.o luacommon.o llg.o spinoperationappliedfield.o spinoperationanisotropy.o spinoperationthermal.o mersennetwister.o llgquat.o interpolatingfunction.o convert.o spinoperationdipole.o dipolesupport.o random.o isaac.o llgcartesian.o crand.o interpolatingfunction2d.o llgfake.o luampi.o luamigrate.cpp

CFLAGS=-g -O2

HOMELIBLUA=$(shell ls $(HOME)/lua-5.1.4/lib/liblua.a 2> /dev/null | wc -l)

ifeq ($(HOMELIBLUA),1)
	LIBS=-L$(HOME)/lib $(HOME)/lua-5.1.4/lib/liblua.a  -ldl -lfftw3
	INCLUDEPATH=-I$(HOME)/include -I$(HOME)/lua-5.1.4/include
else
	LIBS=-llua5.1 -ldl -lfftw3
endif

ifeq ($(CC),mpicc)
	CXX=mpic++ -D_MPI $(CFLAGS)
	BIN=mpi_maglua
else
	CXX=g++ $(CFLAGS)
	BIN=maglua
endif

all: $(BIN)

mpi:
	make CC=mpicc

install: $(BIN)
	mkdir -p $(HOME)/bin
	cp $(BIN) $(HOME)/bin

uninstall:
	rm -f $(HOME)/bin/$(BIN)

$(BIN): ${OBJECTS}
	${CXX} -o ${BIN} ${OBJECTS} ${LIBS}

main.o : main.cpp main.h info.h
	${CXX} ${INCLUDEPATH} -c main.cpp -o main.o

%.o : %.cpp %.h
	${CXX} ${INCLUDEPATH} -c $*.cpp -o $*.o

info.h: premake.sh
	@ ./premake.sh "$(CXX)"

clena: clean

clean:
	rm -f *~
	rm -f maglua mpi_maglua
	rm -f *.o
	rm -f info.h

pack: clean
	cd .. && tar -czf maglua-`date +"%Y-%m-%d-%H.%M.%S"`.tar.gz maglua
	echo "Created archive: `ls ../maglua*-*-*gz | tail -n 1`"
