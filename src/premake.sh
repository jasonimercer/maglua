#!/bin/bash

# Copyright (C) 2008-2011 Jason Mercer.  All rights reserved.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#
# This script takes information about the current
# version, compiler and compiler options and adds
# them to a header filr
#

REVISION=`svn info | grep Revision | sed s/\:/\:\ \ \ \ /`
REVNUMBER=`svn info | grep Revision | sed s/Revision\:\ //`
AUTHOR=`svn info | grep Author:`
DATE=`svn info | grep Date: | sed s/Date\:/Date\:\ \ /`
COMPILER=`$1 --version | head -n 1`;
COMPILELINE="$1"

cat > info.h <<EOF
const char* __info ="\\
MagLua $REVISION\\n\\
$AUTHOR\\n\\
$DATE\\n\\
Compiler:            $COMPILER\\n\\
Compile Line:        $COMPILELINE\\n\\
";
const char* __rev = "$REVNUMBER";
EOF
