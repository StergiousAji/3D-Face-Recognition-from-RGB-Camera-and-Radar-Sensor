# Soli logger readme

This is a modified version of the original Soli logger application, supporting Linux and OSX (including M1 Macs).

There are 3 options for recording radar data:
 - log to a local file (the default)
 - stream packets over TCP socket (optionally recording to a local file at the same time)
 - stream packets over a ZeroMQ socket (optionally recording to a local file at the same time)

## Share Folders Ubuntu
* Inside VirtualBox VM settings, specify share folder `SHARENAME`.
* Boot up Ubuntu and create a new directory to mount to `HOSTNAME`.
* Run `sudo mount -t vboxsf SHARENAME HOSTNAME`

### Disable Password Prompts
* Run `sudo visudo`
* Modify line `%admin ALL=(ALL) ALL` to `%admin ALL=(ALL) NOPASSWD: ALL`

## Compiling

**Dependencies**: Soli libraries (email Andrew if you don't have these), optionally [cppzmq](https://github.com/zeromq/cppzmq).

**Linux**: run `make simplelogger` to compile the basic version, without ZeroMQ support. [cppzmq](https://github.com/zeromq/cppzmq) is required as a dependency if you want to compile the ZeroMQ version (`sudo apt install libzmq3-dev` on Ubuntu), which can be done with `make simplelogger_zmq`. Or you can simply run `make` to compile both.

**OSX**: run `make -f Makefile.osx`.

## First-time setup

On both Linux and OSX you'll typically need to use `sudo` to run the application due to the permissions on the devices involved. On Linux it's possible to update the permissions to avoid this:
 - create a file called /etc/udev/rules.d/50-ftdi.rules
 - paste in this line: `SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="601c", GROUP="users", MODE="0664"` (can change "users" to any other group your user belongs to)
 - save the file, then run `sudo udevadm control --reload-rules && sudo udevadm trigger`

After this you should be able to run the logger without `sudo`. 

## Usage

The simplest case is to record some data to a file:

> sudo ./simplelogger -f somedata

and hit Ctrl-C when you want to stop logging. This will create a file called `somedata_ddmmyyy_hhmmss.radar` in the current directory containing all the radar burst data. 

There are some other command-line options available, view these by running `./simplelogger -h`

Some other examples:

1. Logging for 10s then automatically stopping:

> sudo ./simplelogger -f somedata -t 10

2. Wait for a UDP packet on port 12345 before starting to log for 10 seconds (the content of the packet is ignored):

> sudo ./simplelogger -f somedata -t 10 -w 

3. Stream data over TCP:

> sudo ./simplelogger -f somedata -s 

In this case the logger acts as a TCP server listening for connections on port 22388. See the `python/teststream.py` example.

You can also omit the the `-f` parameter in this mode if you don't care about recording the data to a local file.

4. Stream data using ZeroMQ (note this is using the `simplelogger_zmq` executable!):

> sudo ./simplelogger_zmq -f somedata -z

In this mode the logger acts as a ZeroMQ publisher using port 5556. See the `python/testzmq.py` example. 

You can also omit the the `-f` parameter in this mode if you don't care about recording the data to a local file.

5. Change the radar profile. There are currently 3 sets of parameters supplied by Google: default, shortrange, longrange. If you don't select one, default is used:

> sudo ./simplelogger -f somedata -p shortrange 

## Loading/parsing log files

The `python/soli_logging.py` file contains some useful methods for parsing the .radar files the logger creates, or parsing the packets sent over TCP/ZeroMQ. 

### Parsing an entire log file

To parse a full .radar file at once:

```
from soli_logging import SoliLogParser, get_crd_data_bin, plot_crd_data
params, bursts = SoliLogParser().parse_file(filename)
print(params) # contains the radar parameters used to make the recording
print(len(bursts)) # the number of bursts in the recording
# generate CRD data from the bursts
crd_data = get_crd_data_bin(params, bursts)
# plot some CRD frames
plot_crd_data(crd_data, indices=[x for x in range(100, 200, 5)])
```

### Parsing burst-by-burst

If you're using one of the two streaming modes, you'll receive one burst at a time. In this case it's better to use the parse_burst method in the SoliLogParser class. See the `python/teststream.py` and `python/testzmq.py` scripts for examples of this. 
