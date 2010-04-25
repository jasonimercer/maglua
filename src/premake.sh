#!/bin/bash

#
# This script takes information about the current
# version, compiler and compiler options and adds
# them to a header filr
#

REVISION=`svn info | grep Revision | sed s/\:/\:\ \ \ \ /`
AUTHOR=`svn info | grep Author:`
DATE=`svn info | grep Date: | sed s/Date\:/Date\:\ \ /`
COMPILER=`$1 --version | head -n 1`;
COMPILELINE="$1"

cat > info.h <<EOF
const char* __info ="\\
Maglua $REVISION\\n\\
$AUTHOR\\n\\
$DATE\\n\\
Compiler:            $COMPILER\\n\\
Compile Line:        $COMPILELINE\\n\\
";
EOF
