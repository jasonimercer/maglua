OBJECTS=main.o spinsystem.o spinoperation.o spinoperationexchange.o luacommon.o llg.o spinoperationappliedfield.o spinoperationanisotropy.o spinoperationthermal.o mersennetwister.o llgquat.o interpolatingfunction.o convert.o spinoperationdipole.o dipolesupport.o random.o isaac.o llgcartesian.o crand.o interpolatingfunction2d.o llgfake.o llgalign.o luampi.o luamigrate.o encodable.o

CFLAGS=-ansi -O2 -g

HOMELIBLUA=$(shell ls $(HOME)/lua-5.1.4/lib/liblua.a 2> /dev/null | wc -l)

BUILDTYPE= none
BUILDTYPES=mpi serial

ifeq ($(HOMELIBLUA),1)
	LIBS+=-L$(HOME)/lib $(HOME)/lua-5.1.4/lib/liblua.a  -ldl -lfftw3
	INCLUDEPATH=-I$(HOME)/include -I$(HOME)/lua-5.1.4/include
else
	LIBS+=-llua5.1 -ldl -lfftw3
	INCLUDEPATH=-I/usr/include/lua5.1
endif

CXX=$(CXX) 

all: $(BUILDTYPE)

none:
	@echo "Please choose a build type:"
	@echo "    $(BUILDTYPES)"

mpi:
	$(MAKE) maglua_mpi CXX="mpic++ -D_MPI" BIN=maglua_mpi

serial:
	$(MAKE) maglua CXX="g++" BIN=maglua

all: $(BIN)

install: $(BIN)
	mkdir -p $(HOME)/bin
	cp $(BIN) $(HOME)/bin
	cp maglua_mpi $(HOME)/bin 2> /dev/null

uninstall:
	rm -f $(HOME)/bin/$(BIN)
	rm -f $(HOME)/bin/maglua_mpi

$(BIN): ${OBJECTS}
	${CXX} $(CFLAGS) -o ${BIN} ${OBJECTS} ${LIBS}

main.o : main.cpp main.h info.h
	${CXX} $(CFLAGS) ${INCLUDEPATH} -c main.cpp -o main.o

%.o : %.cpp %.h
	${CXX} $(CFLAGS) ${INCLUDEPATH} -c $*.cpp -o $*.o

info.h: premake.sh
	./premake.sh "$(CXX) $(CFLAGS)"

clena: clean

clean:
	rm -f *~
	rm -f *.o
	rm -f info.h

distclean: clean
	rm -f maglua maglua_mpi

pack: distclean
	cd .. && tar -czf maglua-`date +"%Y-%m-%d-%H.%M.%S"`.tar.gz maglua
	echo "Created archive: `ls ../maglua*-*-*gz | tail -n 1`"
