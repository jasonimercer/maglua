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
NVCC=/usr/bin/nvcc

CFLAGS=-ansi -O2 -g
PKG=lua5.1
INCLUDEPATH=`pkg-config --cflags-only-I $(PKG)` -I../core_purecuda
LIBS=`pkg-config --libs $(PKG) ` -lpthread -lrt -lcudart -lcuda
OBJECTS=spinoperationdipole.o dipolesupport.o kernel.cuo
LIBNAME=PureCUDADipole
MODDIR=$(shell maglua --module_path)

all: $(LIBNAME).so

$(LIBNAME).so : $(OBJECTS)
	$(CXX) -rdynamic -fPIC -shared $(LIBS) $(OBJECTS) -o $(LIBNAME).so

%.o : %.cpp  %.h
	$(CXX) -fPIC $(INCLUDEPATH) -c $*.cpp -o $*.o 

%.cuo : %.cu
	$(NVCC) --ptxas-options="-v" -Xcompiler -fPIC $(INCLUDEPATH) -arch=sm_23 -c $*.cu -o $*.cuo

install: all
	mkdir -p $(MODDIR)
	cp -f $(LIBNAME).so $(MODDIR)

uninstall:
	rm -f $(MODDIR)/$(LIBNAME).so

clena: clean

clean:
	rm -f *~
	rm -f $(OBJECTS)

distclean: clean
	rm -f $(LIBNAME).so



