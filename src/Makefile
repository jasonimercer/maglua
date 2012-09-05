# Copyright (C) 2011 Jason Mercer.  All rights reserved.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# When configuring fftw3: 
# ./configure  --prefix=$HOME --with-pic --enable-shared

# MPICXX=mpicxx -D_MPI
# CXX=g++

BASEDIR=.
include makefile.common.mpi
CFLAGS=-O2 -g
PKG=lua5.1 fftw3 fftw3f
INCLUDEPATH=`pkg-config --cflags-only-I $(PKG)`
LIBS=`pkg-config --libs $(PKG)` -lpthread
OBJECTS=libMagLua.o loader.o modules.o
LIBNAME=libMagLua.a
BIN=maglua


$(BIN): main.o $(LIBNAME)
	$(MPICXX) $(CFLAGS) -o $(BIN) -Wl,--start-group  main.o $(LIBNAME) $(LIBS) -Wl,--end-group

$(LIBNAME) : $(OBJECTS)
	ar cqs $(LIBNAME) $(OBJECTS)

%.o : %.cpp  %.h info.h Makefile bootstrap.h
	$(CXX) -fPIC $(INCLUDEPATH) $(CFLAGS) -c $*.cpp -o $*.o 

bootstrap.h: bootstrap.lua
	lua hardcode.lua bootstrap.lua

info.h:					#dev version
	lua premake.lua $(CXX) $(CFLAGS)	#dev version


install: $(BIN)
	mkdir -p $(HOME)/bin
	cp -f maglua $(HOME)/bin 2> /dev/null || true

uninstall: distclean
	rm -f ~/bin/maglua

clena: clean

clean:
	rm -f *~
	rm -f *.o
	rm -f $(LIBNAME)

pack:			#dev version
	./pack.sh	#dev version

pack_cpu:			#dev version
	./pack_cpu.sh	#dev version

distclean: clean
	make -C modules/cuda distclean #cuda version
	make -C modules/cpu distclean
	make -C modules/common distclean
	make -C modules/extras distclean
	rm -f $(BIN)
	rm -f info.h	#dev version

