#!/bin/bash

# Set the following 2 variables with compilers for your system
# the table below can help with these choices

MPI_COMPILER=
MAGLUA_COMPILER=

# The following table has info for common machines and setups at MUN
# Older versions of g++ will not work: need g++ 4.5 or greater
#
#                       MAGLUA_COMPILER        MPI_COMPILER
# 
#   Modern Linux            g++               mpicxx or mpic++
# g++ 4.5 or better
#
#     Torngat               icc                  mpiicc
#
#
#    Placentia             pgcpp      /usr/local/openmpi.pgi-1.4.4/bin/mpicxx
#


if [ -z $MPI_COMPILER ]; then
	echo "Set the compiler variables in this script and rerun"
	exit 1
fi
if [ -z $MAGLUA_COMPILER ]; then
	echo "Set the compiler variables in this script and rerun"
	exit 1
fi

# don't want to bother looking for magic to tell if
# portlan group has optimizations turned on
if [ "$MAGLUA_COMPILER" = "pgcpp" ]; then
    MAGLUA_COMPILER="$MAGLUA_COMPILER -D__OPTIMIZE__"
    MPI_COMPILER="$MPI_COMPILER -D__OPTIMIZE__"
fi


#  Set BUILD_STEP to restart a failed build
#
#  0   fetch deps, build deps, build maglua
#  1   build deps, build maglua
#  2   build fftw, build maglua
#  3   build maglua

BUILD_STEP=0


# updating info.h with compiler info
cp info.h info.base
head -n 5 info.base > info.h
# portland group has an odd version reporting method 
# so we have to deal with it specially
if [ "$MAGLUA_COMPILER" = "pgcpp" ]; then
	echo "Compiler:            `$MAGLUA_COMPILER --version | head -n 2 | tail -n 1`\n\\" >> info.h
else
	echo "Compiler:            `$MAGLUA_COMPILER --version | head -n 1`\n\\" >> info.h
fi
tail -n 4 info.base >> info.h
rm -f info.base


#adding custom define to compiler cmd
MPI_COMPILER="$MPI_COMPILER -D_MPI"

if [[ $BUILD_STEP -le 0 ]]; then
  echo "#####################################################"
  echo "##############   Fetching dependencies  #############"
  echo "#####################################################"
  if [ ! -f lua-5.1.5.tar.gz ]; then
    wget http://www.lua.org/ftp/lua-5.1.5.tar.gz
  fi
  if [ ! -f fftw-3.3.2.tar.gz ]; then
    wget http://www.fftw.org/fftw-3.3.2.tar.gz
  fi
fi

mkdir -p deps/lib/pkgconfig
cd deps
DEPDIR=`pwd`
cd ..

if [[ $BUILD_STEP -le 1 ]]; then
echo "#####################################################"
echo "###  Building Lua deps and installing in \$DEPDIR  ##"
echo "#####################################################"
echo ""
rm -rf lua-5.1.5
tar -xzf lua-5.1.5.tar.gz
cd lua-5.1.5/src
make all MYCFLAGS="-fPIC -DLUA_USE_LINUX" MYLIBS="-Wl,-E -ldl -lreadline -lhistory -lncurses"
rm -f lua.o luac.o print.o
g++ -shared -o liblua.so *.o
cd ..
make local
mkdir -p ~/bin
rm -f ~/bin/lua
rm -f ~/bin/luac
cp bin/* ~/bin
cp src/liblua.so $DEPDIR/lib
cp -r include $DEPDIR
#cp etc/lua.pc $DEPDIR/lib/pkgconfig
cd ..
fi

#build lua.pc & symlinks
#echo "Building Lua pkgconfig file"
head -n 10 lua-5.1.5/etc/lua.pc >  $DEPDIR/lib/pkgconfig/lua.pc
echo "prefix= $DEPDIR"          >> $DEPDIR/lib/pkgconfig/lua.pc
tail -n 20 lua-5.1.5/etc/lua.pc >> $DEPDIR/lib/pkgconfig/lua.pc

rm -f $DEPDIR/lib/pkgconfig/lua5.1.pc
ln -s $DEPDIR/lib/pkgconfig/lua.pc $DEPDIR/lib/pkgconfig/lua5.1.pc



if [[ $BUILD_STEP -le 2 ]]; then
echo "#####################################################"
echo "###############  Building 64-bit fftw ###############"
echo "#####################################################"
rm -rf fftw-3.3.2
tar -xzf fftw-3.3.2.tar.gz
cd fftw-3.3.2
./configure  --prefix=$DEPDIR --with-pic --enable-shared
make -j10
make install
cd ..
rm -rf fftw-3.3.2

echo "#####################################################"
echo "###############  Building 32-bit fftw ###############"
echo "#####################################################"
tar -xzf fftw-3.3.2.tar.gz
cd fftw-3.3.2
./configure  --prefix=$DEPDIR --with-pic --enable-shared --enable-single
make -j10
make install
cd ..
rm -rf fftw-3.3.2
fi

#echo "Setting temporary environment variables"
export PATH=$HOME/bin:$PATH
export PKG_CONFIG_PATH=$DEPDIR/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$DEPDIR/lib:$LD_LIBRARY_PATH

#pkg-config  --list-all


echo "#####################################################"
echo "#################  Building MagLua ##################"
echo "#####################################################"

make MPICXX="$MPI_COMPILER -D_MPI" CXX="$MAGLUA_COMPILER"
make install

rm -rf libs
mkdir -p libs
cd libs
maglua --setup `pwd`
cd ..

cd modules/common/mpi
make install MPICXX="$MPI_COMPILER -D_MPI" CXX="$MPI_COMPILER -D_MPI"
cd ../../..

cd modules/common
make install MPICXX="$MPI_COMPILER -D_MPI" CXX="$MAGLUA_COMPILER"
cd ../..

cd modules/cpu
make install MPICXX="$MPI_COMPILER -D_MPI" CXX="$MAGLUA_COMPILER"
cd ../..




cd help
maglua help.lua
echo "#####################################################"
echo "######### Help Documentation Generated   ############"
echo "#####################################################"
echo -e "\nHelp file named \"maglua.html\" in:\n     `pwd`\n"
cd ..




echo -e "\n\n"
echo "#####################################################"
echo "###############   Install  Complete   ###############"
echo "#####################################################"
echo -e "\nAdd the following to ~/.bashrc and either source it\nor log out and back in:\n"

cat <<EOF
export PATH=\$HOME/bin:\$PATH
export PKG_CONFIG_PATH=$DEPDIR/lib/pkgconfig:\$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$DEPDIR/lib:\$LD_LIBRARY_PATH
EOF

echo -e ""
echo "#####################################################"


