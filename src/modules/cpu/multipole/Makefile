# Copyright (C) 2012 Jason Mercer.  All rights reserved.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

OBJECTS=spinoperationmultipole.o fmm_math.o plm.o
LIBNAME=Multipole
CATEGORY=cpu

#DEPEND is parsed to make the vs2010 project
DEPEND=core luabaseobject
BASEDIR=../../..
include $(BASEDIR)/makefile.common.cpu


all: $(LIBNAME).so

$(LIBNAME).so : $(OBJECTS)
	$(CXX) $(CFLAGS) -fPIC -rdynamic -shared $(OBJECTS) $(LIBS) $(LIBMIGRATE) -o $(LIBNAME).so

PKG=lua5.1 fftw3
INCLUDEPATH=`pkg-config --cflags-only-I $(PKG)`
LIBS=`pkg-config --libs $(PKG)` -lpthread

main.o: main.cpp
	$(CXX)  $(CFLAGS) -fPIC $(CC_EXTRAOPS) $(INCLUDEPATH) -c main.cpp -o main.o

test: main.o fmm_math.o plm.o
	g++ $(LIBS) fmm_math.o plm.o main.o
