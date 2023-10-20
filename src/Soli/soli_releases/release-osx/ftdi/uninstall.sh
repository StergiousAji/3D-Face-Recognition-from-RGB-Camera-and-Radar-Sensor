#!/bin/bash

#set -x
set -e

CUR_DIR=./
INC_PATH=/usr/local/include
LIB_PATH=/usr/local/lib

echo "Removing header files ..."
sudo rm -rf $INC_PATH/ftd2xx.h  $INC_PATH/libft4222.h $INC_PATH/WinTypes.h
echo "Removing header files done"

echo "Removing libraries ..."
sudo rm -rf $LIB_PATH/libft4222.*
sudo rm -rf $LIB_PATH/libftd2xx.*
echo "Removing libraries done"

