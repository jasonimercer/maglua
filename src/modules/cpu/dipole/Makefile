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

OBJECTS=extrapolate.o spinoperationdipole.o dipolesupport.o
LIBNAME=Dipole
CATEGORY=cpu
EXTRA_CPU_DEPS=extrapolate.h

#DEPEND is parsed to make the vs2010 project
DEPEND=core luabaseobject longrange array
BASEDIR=../../..

EXTRA_CLEAN=extrapolate.cpp extrapolate.h
include $(BASEDIR)/makefile.common.cpu

extrapolate.h: extrapolate.lua $(BASEDIR)/hardcode.lua
	lua $(BASEDIR)/hardcode.lua extrapolate.lua

extrapolate.cpp: extrapolate.lua
	touch extrapolate.cpp
