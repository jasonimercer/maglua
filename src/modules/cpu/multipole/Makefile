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

OBJECTS=fmm_octtree_luafuncs.o spinoperationmultipole.o fmm_math.o plm.o fmm_octtree.o
LIBNAME=Multipole
CATEGORY=cpu

#DEPEND is parsed to make the vs2010 project
DEPEND=core luabaseobject array
BASEDIR=../../..
include $(BASEDIR)/makefile.common.cpu

fmm_octtree_luafuncs.h: fmm_octtree_luafuncs.lua $(BASEDIR)/hardcode.lua
	lua $(BASEDIR)/hardcode.lua fmm_octtree_luafuncs.lua

fmm_octtree_luafuncs.cpp:
	touch fmm_octtree_luafuncs.cpp

EXTRA_CLEAN=fmm_octtree_luafuncs.cpp fmm_octtree_luafuncs.h