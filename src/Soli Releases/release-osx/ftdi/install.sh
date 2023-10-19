#!/bin/bash

#set -x
set -e

CUR_DIR=./
INC_PATH=/usr/local/include
LIB_PATH=/usr/local/lib

echo "Install header files ..."
mkdir -p $INC_PATH
sudo chown -R $(whoami) $INC_PATH
cp $CUR_DIR/inc/*.h $INC_PATH
echo "Install header files done"

echo "Install libraries ..."
mkdir -p $LIB_PATH
sudo chown -R $(whoami) $LIB_PATH
cp $CUR_DIR/lib/*.dylib $LIB_PATH
echo "Install libraries done"

echo "Make symbolic links ..."
ln -sf $LIB_PATH/libft4222.1.4.4.14.dylib $LIB_PATH/libft4222.dylib
ln -sf $LIB_PATH/libftd2xx.1.4.16.dylib $LIB_PATH/libftd2xx.dylib
echo "Make symbolic links done"

