#!/bin/bash
#set -x
set -e

CUR_DIR=./
INC_PATH=/usr/local/include
LIB_PATH=/usr/local/lib

echo "Removing headers ..."
sudo rm -rf $INC_PATH/MustangClient*.h
sudo rm -rf $INC_PATH/MustangSensorParams.h
echo "Removing headers done"

echo "Removing libraries ..."
sudo rm -rf $LIB_PATH/libmc.so
sudo rm -rf $LIB_PATH/libromecontrol.so
echo "Removing libraries done"

