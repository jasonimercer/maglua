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
CXX=g++
NVCC=/usr/bin/nvcc 
NVFLAGS=-O2 --ptxas-options="-v" -Xcompiler -fPIC  -arch=sm_21
CFLAGS=-ansi -O2
PKG=lua5.1
INCLUDEPATH=`pkg-config --cflags-only-I $(PKG)`
LIBS=`pkg-config --libs $(PKG)` -lpthread -lcudart -lcuda
MODDIR=$(shell maglua --module_path)

OBJECTS=spinsystem.o spinoperation.o  luacommon.o \
	mersennetwister.o  interpolatingfunction.o \
	random.o isaac.o crand.o interpolatingfunction2d.o \
	 luamigrate.o encodable.o maglua.o jthread.o \
	 llg.o llgquat.o  spinoperationanisotropy.o \
	spinoperationthermal.o spinoperationexchange.o \
	spinoperationappliedfield.o \
	\
	spinsystem.cuo llgquat.cuo  spinoperationanisotropy.cuo \
	hybridtaus.cuo spinoperationthermal.cuo \
	spinoperationexchange.cuo

TODO= llgcartesian.o llgfake.o llgalign.o
 
LIBNAME=PureCUDACore

all: $(LIBNAME).so

$(LIBNAME).so : $(OBJECTS)
	$(CXX) $(LIBS) -fPIC -rdynamic -shared $(OBJECTS) $(LIBMIGRATE) -o $(LIBNAME).so

%.o : %.cpp  %.h
	$(CXX) -fPIC $(INCLUDEPATH) -c $*.cpp -o $*.o 

%.cuo : %.cu
	$(NVCC) $(NVFLAGS)  $(INCLUDEPATH) -c $*.cu -o $*.cuo

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
