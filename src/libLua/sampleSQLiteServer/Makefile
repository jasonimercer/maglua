# Copyright (C) 2008-2010 Jason Mercer.  All rights reserved.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

OBJECTS=main.o

CFLAGS=-ansi -O2 -g

LIBBASE=..
LUALIBS=$(LIBBASE)/libLuaClient/libLuaClient.a \
        $(LIBBASE)/libLuaServer/libLuaServer.a \
	$(LIBBASE)/libLuaSqlite/libLuaSqlite.a \
	$(LIBBASE)/libLuaMigrate/libLuaMigrate.a

LIBS = `pkg-config --libs lua5.1 sqlite3`  $(LUALIBS)

INCLUDEPATH = `pkg-config --cflags-only-I lua5.1 sqlite3` \
	-I$(LIBBASE)/libLuaClient \
	-I$(LIBBASE)/libLuaSqlite \
	-I$(LIBBASE)/libLuaServer \
	-I$(LIBBASE)/libLuaMigrate

BIN=luasql_server
CXX=g++

all: $(BIN)

test: all

$(BIN): $(OBJECTS) $(LUALIBS)
	$(CXX) $(CFLAGS) -o ${BIN} ${OBJECTS} ${LIBS}

%.o : %.cpp %.h
	$(CXX) $(CFLAGS) ${INCLUDEPATH} -c $*.cpp -o $*.o

clena: clean

clean:
	rm -f *~
	rm -f *.o

distclean: clean
	rm -f $(BIN)
