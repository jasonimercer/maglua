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

OBJECTS=spinoperationlongrange.o longrange_kernel.cuo longrange.o
LIBNAME=LongRangeCuda
CATEGORY=cuda32

#DEPEND is parsed to make the vs2010 project
DEPEND=luabaseobject corecuda
BASEDIR=../../..
EXTRALIBS=-lcudart -lcuda
EXTRA_CUDA_DEP=longrange.h

include $(BASEDIR)/makefile.common.cpu
include $(BASEDIR)/makefile.common.gpu32

longrange.h: longrange.lua $(BASEDIR)/hardcode.lua
	lua $(BASEDIR)/hardcode.lua longrange.lua

longrange.cpp: longrange.lua
	touch longrange.cpp
