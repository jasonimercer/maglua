# Copyright (C) 2011 Jason Mercer.  All rights reserved.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

CXX=g++

CFLAGS=-ansi -O2 -g
# CFLAGS=-ansi -g
PKG=lua5.1 fftw3
INCLUDEPATH=`pkg-config --cflags-only-I $(PKG) `
LIBS=`pkg-config --libs $(PKG)` -lpthread

OBJECTS=spinsystem.o spinoperation.o spinoperationexchange.o luacommon.o \
	llg.o spinoperationappliedfield.o spinoperationanisotropy.o spinoperationthermal.o \
	mersennetwister.o llgquat.o interpolatingfunction.o spinoperationdipole.o \
	dipolesupport.o random.o isaac.o llgcartesian.o crand.o interpolatingfunction2d.o \
	llgfake.o llgalign.o luampi.o luamigrate.o encodable.o  maglua.o

DEFINES=-DMAGLUAMODULESPATH=\"$(shell pwd)/lib\"
LIBNAME=Core
BIN=maglua

all: lib bin

bin: main.o
	$(CXX) $(CFLAGS) -o $(BIN) main.o $(LIBS)

lib: $(LIBNAME).so

$(LIBNAME).so : $(OBJECTS)
	$(CXX) -fPIC -shared $(OBJECTS) $(LIBMIGRATE) -o $(LIBNAME).so

main.o : main.cpp  info.h
	$(CXX) -fPIC  $(DEFINES) $(INCLUDEPATH) -c $*.cpp -o $*.o 

%.o : %.cpp  %.h
	$(CXX) -fPIC $(INCLUDEPATH) -c $*.cpp -o $*.o 

info.h: premake.sh                         # working SVN required
	@ ./premake.sh "$(CXX) $(CFLAGS)"  # working SVN required
	@ echo "premake"                   # working SVN required

install: all
	mkdir -p lib
	mv -f $(LIBNAME).so lib
	mkdir -p $(HOME)/bin
	cp -f maglua $(HOME)/bin 2> /dev/null || true
	cp -f maglua_mpi $(HOME)/bin 2> /dev/null || true
	cp -f maglua_openmp $(HOME)/bin 2> /dev/null || true


uninstall: distclean
	rm -f $(LIBNAME).so

clena: clean

clean:
	rm -f *~
	rm -f *.o

distclean: clean
	rm -f $(LIBNAME).a
	rm -f $(LIBNAME).so
	rm -f $(BIN)


# 
# 
# 
# # Copyright (C) 2008-2010 Jason Mercer.  All rights reserved.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# # EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# # MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# # IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# # CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# # TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# # SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 
# # SVN is a keyword in this Makefile that is searched for in pack:
# OBJECTS=spinsystem.o spinoperation.o spinoperationexchange.o luacommon.o \
# 	llg.o spinoperationappliedfield.o spinoperationanisotropy.o spinoperationthermal.o \
# 	mersennetwister.o llgquat.o interpolatingfunction.o spinoperationdipole.o \
# 	dipolesupport.o random.o isaac.o llgcartesian.o crand.o interpolatingfunction2d.o \
# 	llgfake.o llgalign.o luampi.o luamigrate.o encodable.o  spinoperationdipoledisordered.o
# 
# #	spinoperationmagnetostatics.o magnetostaticssupport.o
# 
# CFLAGS=-ansi -O2 -g
# 
# BUILDTYPE=serial
# BUILDTYPES=mpi serial openmp
# 
# #these lua libs provide client and db
# # LIBBASE=libLua
# # LUALIBS=$(LIBBASE)/libLuaMigrate/libLuaMigrate.a
# 
# #	$(LIBBASE)/libLuaSqlite/libLuaSqlite.a \
# 
# LIBNAME=libMaglua
# 
# # This makefile uses pkg-config. If you do not have Lua and FFTW registered 
# # with pkg-config, please fill out the proper LIBS and INCLUDEPATH
# PKGS=lua5.1 fftw3 
# #sqlite3
# LIBS = `pkg-config --libs $(PKGS)` 
# # magnetostatics/magnetostatics.a
# INCLUDEPATH = `pkg-config --cflags-only-I $(PKGS)` 
# # 		-IlibLua/libLuaSqlite  \
# # 		-IlibLua/libLuaMigrate \
# # 		-IlibLua/libLuaClient 
# DEFINES=-DMAGLUAMODULESPATH=\"$(shell pwd)/lib\"
# 
# all: $(BUILDTYPE)
# 
# # $(LIBNAME) : $(OBJECTS)
# # 	ar cqs $(LIBNAME) $(OBJECTS)
# 
# $(LIBNAME).so : $(OBJECTS)
# 	$(CXX) -Wl,-rpath,$(shell pwd) -fPIC -shared $(LIBS) $(OBJECTS) -o $(LIBNAME).so
# none:
# 	@echo "Please choose a build type:"
# 	@echo "    $(BUILDTYPES)"
# 
# mpi:
# 	@ $(MAKE) maglua_mpi CXX="mpic++ -D_MPI" BIN=maglua_mpi
# 
# serial:
# 	@ $(MAKE) maglua CXX="g++" BIN=maglua
# 
# openmp:
# # 	@ $(MAKE) maglua_openmp CXX="kinst-ompp g++ -fopenmp" BIN=maglua_openmp
# 	@ $(MAKE) maglua_openmp CXX="g++ -fopenmp" BIN=maglua_openmp
# 
# all: $(BIN)
# 
# install: $(BIN)
# 	@ mkdir -p $(HOME)/bin
# 	@ cp -f maglua $(HOME)/bin 2> /dev/null || true
# 	@ cp -f maglua_mpi $(HOME)/bin 2> /dev/null || true
# 	@ cp -f maglua_openmp $(HOME)/bin 2> /dev/null || true
# 	@ echo "Copying binaries to ~/bin"
# 
# uninstall:
# 	rm -f $(HOME)/bin/$(BIN)
# 	rm -f $(HOME)/bin/maglua_mpi
# 
# $(BIN): main.o $(LIBNAME).so
# 	${CXX} -Wl,-rpath,$(shell pwd) $(CFLAGS) -o ${BIN} -L. -lMaglua main.o  ${LIBS}
# 	echo "LD ${BIN}"
# 
# main.o : main.cpp main.h info.h
# 	@ ${CXX} $(CFLAGS) $(DEFINES) ${INCLUDEPATH} -fPIC -c main.cpp -o main.o
# 	@ echo "CXX main"
# 
# %.o : %.cpp %.h
# 	@ ${CXX} $(CFLAGS) $(DEFINES) ${INCLUDEPATH} -fPIC -c $*.cpp -o $*.o
# 	@ echo "CXX $*"
# 
# info.h: premake.sh                         # working SVN required
# 	@ ./premake.sh "$(CXX) $(CFLAGS)"  # working SVN required
# 	@ echo "premake"                   # working SVN required
# 
# clena: clean
# 
# clean:
# 	@ rm -f *~
# 	@ rm -f *.o
# 	@ rm -f info.h # working SVN needed to rebuild
# 	@ rm -f rev
# 	@ rm -f *.inc *.mod.* #ompp profiling files
# 	@ echo "Temporary files removed"
# 
# distclean: clean
# 	make -C libLua clean
# #	make -C magnetostatics clean
# 	rm -f maglua maglua_mpi maglua_openmp
# 
# pack:   info.h
# 	echo maglua-r`cat info.h | grep Revision | sed -e 's/.*\s\([0-9]\{1,\}\).*/\1/'` > rev
# 	rm -rf `cat rev`
# 	mkdir `cat rev`
# 	cp *.cpp *.h help.lua maglua.css README COPYRIGHT `cat rev`
# 	cp -r ../doc/manual `cat rev`
# #	cp -r magnetostatics `cat rev`
# 	cp -r libLua `cat rev`
# 	cat Makefile | head -n -10 | grep -v SVN > `cat rev`/Makefile
# 	tar -czf `cat rev`.tar.gz `cat rev`
# 	@echo "Created archive: `cat rev`.tar.gz"
# 	rm -f rev
