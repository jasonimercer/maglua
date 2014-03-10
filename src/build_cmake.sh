#!/bin/bash

BUILD_CMAKE=1

if [[ $BUILD_CMAKE = 1 ]]; then
echo "#####################################################"
echo "##################  Building cmake ##################"
echo "#####################################################"
if [ ! -f cmake-2.8.12.2.tar.gz ]; then
  wget http://www.cmake.org/files/v2.8/cmake-2.8.12.2.tar.gz
fi
rm -rf cmake-2.8.12.2
tar -xzf cmake-2.8.12.2.tar.gz
cd cmake-2.8.12.2
./configure  --prefix=$HOME
make -j10
make install
cd ..
rm -rf cmake-2.8.12.2
fi




if [[ $BUILD_CMAKE = 1 ]]; then
echo "You installed cmake, this means you need to add the following to the end of your ~/.bashrc:"
cat <<EOF
export CMAKE_ROOT=\$HOME/share/cmake-2.8
EOF
fi


