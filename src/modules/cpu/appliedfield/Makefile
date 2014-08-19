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

OBJECTS=appliedfield_site_luafuncs.lo appliedfield_luafuncs.lo appliedfield_heterogeneous_luafuncs.lo spinoperationappliedfield.o spinoperationappliedfield_heterogeneous.o spinoperationappliedfield_site.o
LIBNAME=AppliedField
CATEGORY=cpu

#DEPEND is parsed to make the vs2010 project
DEPEND=core luabaseobject array
BASEDIR=../../..
include $(BASEDIR)/makefile.common.cpu

EXTRA_CLEAN=appliedfield_luafuncs.cpp appliedfield_luafuncs.h appliedfield_heterogeneous_luafuncs.cpp appliedfield_heterogeneous_luafuncs.h appliedfield_site_luafuncs.cpp appliedfield_site_luafuncs.h

