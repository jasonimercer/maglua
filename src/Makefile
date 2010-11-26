# Copyright (C) 2008-2010 Jason Mercer.  All rights reserved.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# SVN is a keyword in this Makefile that is searched for in pack:
OBJECTS=main.o spinsystem.o spinoperation.o spinoperationexchange.o luacommon.o \
	llg.o spinoperationappliedfield.o spinoperationanisotropy.o spinoperationthermal.o \
	mersennetwister.o llgquat.o interpolatingfunction.o spinoperationdipole.o \
	dipolesupport.o random.o isaac.o llgcartesian.o crand.o interpolatingfunction2d.o \
	llgfake.o llgalign.o luampi.o luamigrate.o encodable.o  spinoperationdipoledisordered.o 

CFLAGS=-ansi -O2 -g

BUILDTYPE=none
BUILDTYPES=mpi serial openmp

# This makefile uses pkg-config. If you do not have Lua and FFTW registered 
# with pkg-config, please fill out the proper LIBS and INCLUDEPATH
LIBS = `pkg-config --libs lua5.1 fftw3`
INCLUDEPATH = `pkg-config --cflags-only-I lua5.1 fftw3`

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
	@ rm -f rev
	@ echo "Temporary files removed"

distclean: clean
	rm -f maglua maglua_mpi maglua_openmp

pack:   info.h
	echo maglua-r`cat info.h | grep Revision | sed -e 's/.*\s\([0-9]\{1,\}\).*/\1/'` > rev
	rm -rf `cat rev`
	mkdir `cat rev`
	cp *.cpp *.h help.lua maglua.css README COPYRIGHT `cat rev`
	cp -r ../doc/manual `cat rev`
	cat Makefile | head -n -10 | grep -v SVN > `cat rev`/Makefile
	tar -czf `cat rev`.tar.gz `cat rev`
	@echo "Created archive: `cat rev`.tar.gz"
	rm -f rev
