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


# The following rules are used to make a "lua object" which is a lua script 
# hardcoded and compiled

# This secondary tag keeps intermediate files, which we want
.SECONDARY:

%.h: %.lua $(BASEDIR)/hardcode.lua
	 lua $(BASEDIR)/hardcode.lua $*.lua

%.cpp: %.lua $(BASEDIR)/hardcode.lua
	 lua $(BASEDIR)/hardcode.lua $*.lua

%.lo : %.h %.cpp  %.lua $(EXTRA_CPU_DEPS)
	 $(CXX)  $(CFLAGS) -fPIC $(CC_EXTRAOPS) $(INCLUDEPATH) -c $*.cpp -o $*.lo
