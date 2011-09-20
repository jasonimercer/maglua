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
#
# When configuring fftw3:
# ./configure  --prefix=$(HOME) --with-pic --enable-shared

#
#CXX=g++
CXX=mpic++ -D_MPI

CFLAGS=-ansi -O2

PKG=lua5.1 fftw3
INCLUDEPATH=`pkg-config --cflags-only-I $(PKG)`
LIBS=`pkg-config --libs $(PKG)` -lpthread

BIN=maglua

.PHONY : mpi


all: $(BIN)

mpi:
	make CXX="mpic++ -D_MPI" BIN=maglua_mpi all

$(BIN): main.o
	$(CXX) -rdynamic  $(CFLAGS) -o $(BIN) -Wl,--start-group $(LIBS) main.o  -Wl,--end-group

# lib/Core.so:
# 	make -C core install

main.o : main.cpp  info.h setup_code.h
	$(CXX) -fPIC $(INCLUDEPATH) -c $*.cpp -o $*.o 

install: all
	mkdir -p $(HOME)/bin
	cp -f maglua $(HOME)/bin 2> /dev/null || true

info.h:
	./premake.sh $(CXX) $(CFLAGS)

uninstall: distclean
	rm -f ~/bin/maglua

clena: clean

clean:
	rm -f *~
	rm -f *.o

distclean: clean
	rm -f $(BIN)
	rm -f info.h
