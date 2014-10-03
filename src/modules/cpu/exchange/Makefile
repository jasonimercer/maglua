# Copyright (C) 2014 Jason Mercer.  All rights reserved.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

OBJECTS=exchange_main.lo exchange_luafuncs.lo spinoperationexchange.o
LIBNAME=Exchange
CATEGORY=cpu

# If you understand the implications of the change to the exchange operator
# then you can uncomment the following line:
# EXTRA_CFLAGS=-D_UNDERSTAND_EXCHANGE_CHANGE


#DEPEND is parsed to make the vs2010 project
DEPEND=core luabaseobject array
BASEDIR=../../..
include $(BASEDIR)/makefile.common.cpu
EXTRA_CLEAN=exchange_luafuncs.h exchange_luafuncs.cpp  exchange_main.h exchange_main.cpp


