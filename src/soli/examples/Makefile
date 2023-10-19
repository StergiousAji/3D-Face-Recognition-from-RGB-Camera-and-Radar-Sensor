# Makefile for a hello world Soli example.
TARGET = main

all: $(TARGET)

$(TARGET):
	g++ -std=c++11 -Wall main.cc -I/usr/local/include -L. -lsoli -lmc -v -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(TARGET)

.PHONY: clean run
