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

OBJECTS=mag2d_luafuncs.lo pointfunc_demag.o demag.o spinoperationmagnetostatics2d.o
LIBNAME=Magnetostatics2D
CATEGORY=cpu

#DEPEND is parsed to make the vs2010 project
DEPEND=core dipole luabaseobject longrange2d
BASEDIR=../../..


EXTRA_CLEAN=mag2d_luafuncs.cpp mag2d_luafuncs.h
include $(BASEDIR)/makefile.common.cpu

