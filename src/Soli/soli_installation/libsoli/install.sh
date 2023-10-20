#!/bin/bash
#set -x
set -e

CUR_DIR=./
INC_PATH=/usr/local/include
LIB_PATH=/usr/local/lib
UNAME=$(uname -s)

echo "Install header files ..."
mkdir -p $INC_PATH
sudo chown -R $(whoami) $INC_PATH
cp $CUR_DIR/inc/*.h $INC_PATH
echo "Install header files done"

echo "Install libraries ..."
mkdir -p $LIB_PATH
sudo chown -R $(whoami) $LIB_PATH
cp $CUR_DIR/lib/*.a $LIB_PATH

if [ $UNAME == "Linux" ]; then
    sudo ldconfig $LIB_PATH
fi

echo "Install libraries done"

