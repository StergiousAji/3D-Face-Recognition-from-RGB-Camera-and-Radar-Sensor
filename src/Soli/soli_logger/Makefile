TARGET = simplelogger
ZMQ_TARGET = simplelogger_zmq
# Edit these if needed (the defaults should match the locations
# where the Google install.sh script places the files). Note that
# the original install.sh script for the libmc files doesn't 
# copy the header files in libmc/inc to /usr/local/include, you
# will need to do that manually
SOLI_INCLUDES = /usr/local/include
SOLI_LIBS = /usr/local/lib/

all: $(TARGET) ${ZMQ_TARGET}

$(TARGET): simplelogger.cc
	g++ -std=c++11 -Wall simplelogger.cc -I$(SOLI_INCLUDES) -L$(SOLI_LIBS) -Wl,-rpath=$(SOLI_LIBS) -lsoli -lmc -v -o $@

$(ZMQ_TARGET): simplelogger.cc
	g++ -std=c++11 -Wall simplelogger.cc -DZMQ -I$(SOLI_INCLUDES) -L$(SOLI_LIBS) -Wl,-rpath=$(SOLI_LIBS) -lsoli -lmc -lzmq -v -o $@

clean:
	rm -rf $(TARGET) ${ZMQ_TARGET}

.PHONY: clean
