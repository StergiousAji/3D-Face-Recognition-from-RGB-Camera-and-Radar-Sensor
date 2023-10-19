Soli library. Copyright Google 2021. All rights reserved.

Setup steps:

1) Plug in the Soli dev board via USB to your Linux machine.

2) Run the ./install.sh script.

3) cd to the examples directory and make the example:
  cd examples
  make

4) run the example (note, sudo is required to access the driver!!)
  sudo ./main

Directory as of 2021.04.23
Windows 10 host (laptop), Ubuntu 64-bit VM
> dir data
> dir examples
	> dir src
		> radar_utils.py - standard radar preprocessing functions
		> utils.py - functions used across the project
	> data_processing.ipynb: pre-processes raw radar, cleans and resizes image data, and resamples radar frames to match image timestamps
	> main, main.cc, MakeFile: inherited from Google SOLI demo, main.cc edited to acts as a UDP client and to save radar frames into a JSON file.
	> RS.py: controls Intel Realsense D435 camera, and acts as UDP server
> dir ftdi: inherited from Google SOLI demo, Future Technology Devices International library
> dir libmc: inherited from Google SOLI demo, MustangClient library
> dir libsoli: inherited from Google SOLI demo, SOLI library
> install.sh: inherited from Google SOLI demo, shell script installation file
> requirements.txt: python package dependencies
> README.txt



