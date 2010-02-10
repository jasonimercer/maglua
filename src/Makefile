OBJECTS=main.o spinsystem.o spinoperation.o spinoperationexchange.o luacommon.o llg.o spinoperationappliedfield.o spinoperationanisotropy.o spinoperationthermal.o mersennetwister.o llgquat.o interpolatingfunction.o convert.o spinoperationdipole.o dipolesupport.o random.o isaac.o llgcartesian.o crand.o interpolatingfunction2d.o

BIN=maglua
# CXX=g++ -g -O2 -fopenmp
CXX=g++ -g -O2

# LIBS=-llua5.1 -ldl -lfftw3
LIBS=$(HOME)/lua-5.1.4/lib/liblua.a  -ldl -lfftw3
INCLUDEPATH=-I$(HOME)/lua-5.1.4/include
	 
all: info.h $(BIN)

$(BIN): ${OBJECTS}
	${CXX} -o ${BIN} ${OBJECTS} ${LIBS}

%.o : %.cpp %.h
	${CXX} ${INCLUDEPATH} -c $*.cpp -o $*.o

info.h: premake.sh
	./premake.sh

clena: clean

clean:
	rm -f *~
	rm -f ${BIN}
	rm -f *.o
	rm -f info.h

pack: clean
	cd .. && tar -czf maglua-`date +"%Y-%m-%d-%H.%M.%S"`.tar.gz maglua
	echo "Created archive: `ls ../maglua*-*-*gz | tail -n 1`"
