#!/bin/bash

set -e

UNAME=$(uname -s)

echo "Installing radar library ..."
pushd .
cd ./libmc
./install.sh
popd
echo "Installing radar library done"

echo "Installing FTDI library ..."
pushd .
cd ./ftdi
./install.sh
popd
echo "Installing FTDI library done"

echo "Installing Soli library ..."
pushd .
cd ./libsoli
./install.sh
popd
echo "Installing Soli library done"
