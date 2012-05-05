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

OBJECTS=mersennetwister.o random.o isaac.o crand.o random_wrapper.o
LIBNAME=Random
CATEGORY=common

DEPEND=luabaseobject
BASEDIR=../../..
include $(BASEDIR)/makefile.common.cpu

random_wrapper.h: random_wrapper.lua $(BASEDIR)/hardcode.lua
	lua $(BASEDIR)/hardcode.lua random_wrapper.lua

random_wrapper.cpp: random_wrapper.lua
	touch random_wrapper.cpp
