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
cp $CUR_DIR/lib/libft4222.so.1.4.4.9 $LIB_PATH
echo "Install libraries done"

echo "Make symbolic links ..."
ln -sf $LIB_PATH/libft4222.so.1.4.4.9 $LIB_PATH/libft4222.so
sudo ldconfig $LIB_PATH
echo "Make symbolic links done"

