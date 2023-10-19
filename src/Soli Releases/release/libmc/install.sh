#!/bin/bash
#set -x
set -e

CUR_DIR=./
LIB_PATH=/usr/local/lib
UNAME=$(uname -s)

echo "Install libraries ..."
mkdir -p $LIB_PATH
sudo chown -R $(whoami) $LIB_PATH
cp $CUR_DIR/lib/*.so $LIB_PATH

if [ $UNAME == "Linux" ]; then
    sudo ldconfig $LIB_PATH
fi
echo "Install libraries done"

