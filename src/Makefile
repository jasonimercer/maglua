# SVN is a keyword in this Makefile that is searched for in pack:
OBJECTS=main.o spinsystem.o spinoperation.o spinoperationexchange.o luacommon.o \
	llg.o spinoperationappliedfield.o spinoperationanisotropy.o spinoperationthermal.o \
	mersennetwister.o llgquat.o interpolatingfunction.o spinoperationdipole.o \
	dipolesupport.o random.o isaac.o llgcartesian.o crand.o interpolatingfunction2d.o \
	llgfake.o llgalign.o luampi.o luamigrate.o encodable.o

CFLAGS=-ansi -O2 -g

HOMELIBLUA=$(shell ls $(HOME)/lua-5.1.4/lib/liblua.a 2> /dev/null | wc -l)

BUILDTYPE= none
BUILDTYPES=mpi serial openmp

ifeq ($(HOMELIBLUA),1)
	LIBS+=-L$(HOME)/lib $(HOME)/lua-5.1.4/lib/liblua.a  -ldl -lfftw3
	INCLUDEPATH=-I$(HOME)/include -I$(HOME)/lua-5.1.4/include
else
	LIBS+=-llua5.1 -ldl -lfftw3
	INCLUDEPATH=-I/usr/include/lua5.1
endif

all: $(BUILDTYPE)

none:
	@echo "Please choose a build type:"
	@echo "    $(BUILDTYPES)"

mpi:
	@ $(MAKE) maglua_mpi CXX="mpic++ -D_MPI" BIN=maglua_mpi

serial:
	@ $(MAKE) maglua CXX="g++" BIN=maglua

openmp:
	@ $(MAKE) maglua_openmp CXX="g++ -fopenmp" BIN=maglua_openmp
	

all: $(BIN)

install: $(BIN)
	@ mkdir -p $(HOME)/bin
	@ cp -f maglua $(HOME)/bin 2> /dev/null || true
	@ cp -f maglua_mpi $(HOME)/bin 2> /dev/null || true
	@ cp -f maglua_openmp $(HOME)/bin 2> /dev/null || true
	@ echo "Copying binaries to ~/bin"

uninstall:
	rm -f $(HOME)/bin/$(BIN)
	rm -f $(HOME)/bin/maglua_mpi

$(BIN): ${OBJECTS}
	@ ${CXX} $(CFLAGS) -o ${BIN} ${OBJECTS} ${LIBS}
	@ echo "LD ${BIN}"

main.o : main.cpp main.h info.h
	@ ${CXX} $(CFLAGS) ${INCLUDEPATH} -c main.cpp -o main.o
	@ echo "CC main"

%.o : %.cpp %.h
	@ ${CXX} $(CFLAGS) ${INCLUDEPATH} -c $*.cpp -o $*.o
	@ echo "CC $*"

info.h: premake.sh                         # working SVN required
	@ ./premake.sh "$(CXX) $(CFLAGS)"  # working SVN required
	@ echo "premake"                   # working SVN required

clena: clean

clean:
	@ rm -f *~
	@ rm -f *.o
	@ rm -f info.h # working SVN needed to rebuild
	@ echo "Temporary files removed"

distclean: clean
	rm -f maglua maglua_mpi maglua_openmp

pack:   info.h
	echo maglua-r`cat info.h | grep Revision | sed -e 's/.*\s\([0-9]\{1,\}\).*/\1/'` > rev
	rm -rf `cat rev`
	mkdir `cat rev`
	cp *.cpp *.h `cat rev`
	cat Makefile | grep -v SVN > `cat rev`/Makefile
	tar -czf `cat rev`.tar.gz `cat rev`
	echo "Created archive: `cat rev`.tar.gz"
