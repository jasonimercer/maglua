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
PKG=lua5.1 sqlite3
INCLUDEPATH=`pkg-config --cflags-only-I $(PKG) ` -I../libLuaMigrate
LIBS=`pkg-config --libs $(PKG) ` -lpthread
OBJECTS=libLuaSqlite.o

LIBNAME=Sqlite
BIN=sql

all: $(LIBNAME).so

$(LIBNAME).so : $(OBJECTS)
	$(CXX) -fPIC -shared $(LIBS) $(OBJECTS) -o $(LIBNAME).so

%.o : %.cpp  %.h
	$(CXX) -fPIC $(INCLUDEPATH) -c $*.cpp -o $*.o 

test: $(LIBNAME) main.o ../libLuaMigrate/libLuaMigrate.a
	$(CXX) $(CFLAGS) -o $(BIN) main.o $(LIBNAME) $(LIBS)  ../libLuaMigrate/libLuaMigrate.a

install: all
	mkdir -p ../lib
	cp -f $(LIBNAME).so ../lib

uninstall: distclean
	rm -f ../lib/$(LIBNAME).so

clena: clean

clean:
	rm -f *~
	rm -f *.o

distclean: clean
	rm -f $(LIBNAME).a
	rm -f $(LIBNAME).so
	rm -f $(BIN)
