#!/bin/bash

make install
version=`maglua -v`;

echo $version
rm -rf $version
mkdir $version
cp info.h $version
grep -v '#dev version' Makefile > $version/Makefile
cp COPYRIGHT modules.h modules.cpp loader.cpp loader.h import.h main.cpp makefile.common.cpu makefile.common.gpu makefile.common.mpi README main.h $version
make distclean
cp -r help $version
cp -r modules $version
rm -rf `du -a $version | cut -f 2 | grep '\.svn'`

tar -czf $version.tar.gz $version
