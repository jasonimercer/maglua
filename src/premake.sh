#!/bin/bash

REVISION=`svn info | grep Revision`
AUTHOR=`svn info | grep Author:`
DATE=`svn info | grep Date:`

# cat > info.h

cat > info.h <<EOF
const char* __info ="Maglua $REVISION\\n\\
$AUTHOR\\n\\
$DATE\\n";
EOF


# echo $SRC > info.h

