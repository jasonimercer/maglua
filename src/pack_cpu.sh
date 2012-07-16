#!/bin/bash

make install
version=`maglua -v`-cpu;

echo $version
rm -rf $version
mkdir $version
cp info.h $version
grep -v '#dev version' Makefile > pack_makefile_del_this
grep -v '#cuda version' pack_makefile_del_this > $version/Makefile
rm -f pack_makefile_del_this
cp build_cpu.sh libMagLua.h libMagLua.cpp hardcode.lua bootstrap.lua COPYRIGHT modules.h modules.cpp loader.cpp loader.h import.h main.cpp makefile.common.cpu makefile.common.mpi makefile.common.config README main.h $version
make distclean
cp -r help $version

mkdir -p $version/modules/cpu
mkdir -p $version/modules/extras
mkdir -p $version/modules/common

cp    modules/common/Makefile          $version/modules/common
cp -r modules/common/checkpoint        $version/modules/common
cp -r modules/common/luabaseobject     $version/modules/common
cp -r modules/common/interpolate       $version/modules/common
cp -r modules/common/mpi               $version/modules/common
cp -r modules/common/random            $version/modules/common
cp -r modules/common/timer             $version/modules/common

cp    modules/cpu/Makefile             $version/modules/cpu
cp -r modules/cpu/array                $version/modules/cpu
cp -r modules/cpu/anisotropy           $version/modules/cpu
cp -r modules/cpu/appliedfield         $version/modules/cpu
cp -r modules/cpu/core                 $version/modules/cpu
cp -r modules/cpu/dipole               $version/modules/cpu
cp -r modules/cpu/disordereddipole     $version/modules/cpu
cp -r modules/cpu/exchange             $version/modules/cpu
cp -r modules/cpu/llg                  $version/modules/cpu
cp -r modules/cpu/longrange            $version/modules/cpu
cp -r modules/cpu/magnetostatics       $version/modules/cpu
cp -r modules/cpu/thermal              $version/modules/cpu

cp    modules/extras/Makefile          $version/modules/extras
cp -r modules/extras/pngwriter         $version/modules/extras
cp -r modules/extras/sqlite3           $version/modules/extras


rm -rf `du -a $version | cut -f 2 | grep '\.svn'`

tar -czf $version.tar.gz $version