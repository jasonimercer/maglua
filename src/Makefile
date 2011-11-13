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

CXX=mpic++ -D_MPI
CFLAGS=-ansi -O2 -g
PKG=lua5.1 fftw3
INCLUDEPATH=`pkg-config --cflags-only-I $(PKG)`
LIBS=`pkg-config --libs $(PKG)` -lpthread
OBJECTS=main.o loader.o modules.o

BIN=maglua

$(BIN): $(OBJECTS)
	$(CXX) -rdynamic  $(CFLAGS) -o $(BIN) -Wl,--start-group $(LIBS) $(OBJECTS)  -Wl,--end-group

%.o : %.cpp  %.h info.h Makefile
	$(CXX) -fPIC $(INCLUDEPATH) $(CFLAGS) -c $*.cpp -o $*.o 

info.h:					#dev version
	./premake.sh $(CXX) $(CFLAGS)	#dev version


install: $(BIN)
	mkdir -p $(HOME)/bin
	cp -f maglua $(HOME)/bin 2> /dev/null || true

uninstall: distclean
	rm -f ~/bin/maglua

clena: clean

clean:
	rm -f *~
	rm -f *.o

pack:			#dev version
	./pack.sh	#dev version

distclean: clean
	make -C modules/cuda distclean
	make -C modules/cpu distclean
	make -C modules/common distclean
	make -C modules/extras distclean
	rm -f $(BIN)
	rm -f info.h	#dev version

