// A simple Soli example demonstrating on how to get radar data from the sensor.

#include <chrono>
#include <iostream>
#include <thread>
#include <fstream>
#include "soli.h"
#include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#define PORT 8080
#include <sys/time.h>
#include <sys/resource.h>

std::ofstream file;

void realtime_priority(int priority, int schedtype, int niceval)
{
    struct sched_param schedparm;
    memset(&schedparm, 0, sizeof(schedparm));
    schedparm.sched_priority = priority; 
    sched_setscheduler(0, schedtype, &schedparm);
    setpriority(PRIO_PROCESS, 0, niceval);
}



void PrintRadarMetadata(const SoliRadarBurst& burst) {
  if (!burst.is_valid) return;
  const int num_chirps_per_burst = burst.chirp_data.size();
  const int num_rx_channels = burst.chirp_data[0].channel_data.size();
  const int num_samples_per_chirp = burst.chirp_data[0].channel_data[0].size();
  std::cout << "-burst id: " << burst.burst_id << std::endl;
  std::cout << "-DSP timestamp: " << burst.timestamp_ms << std::endl;
  //std::cout << "-num chirps per burst: " << num_chirps_per_burst << std::endl;
  //std::cout << "-num samples per chirp: " << num_samples_per_chirp << std::endl;
  //std::cout << "-num rx channels: " << num_rx_channels << std::endl;
  //Write burst metadata
  file << "{\"burst_id\":" << burst.burst_id << ",";
  file << "\"timestamp_ms\":" << burst.timestamp_ms << ",";

}

void PrintRadarData(const SoliRadarBurst& burst) {
  //file < "";
  if (!burst.is_valid) return;
  // ! std::cout << "-radar data:" << std::endl;
  file << "\"chirps\":[";
  //loop over chirps per burst e.g. 16.
  for (const SoliRadarChirp& chirp_data : burst.chirp_data) {
    // ! std::cout << "timestamp " << chirp_data.timestamp_usec << " ";
    //loop over channels, e.g. 3.
    for (size_t rx_channel = 0; rx_channel < chirp_data.channel_data.size(); rx_channel++) {
      //loop over samples per chirp, .e.g. 64.
      for (const float value : chirp_data.channel_data[rx_channel]) {
        // ! std::cout << value << " ";
        file << value << ",";
        
      }
    }
  }
  file.seekp(-1, std::ios_base::cur);
  file << "]},";
  //std::cout << std::endl;
  //std::cout << burst.temperature_celsius;
  std::cout << std::endl;
}

void BurstDataCallback(const SoliRadarBurst& burst) {
  if (!burst.is_valid) {
    std::cout << "Got radar data, but it is invalid!!" << std::endl;
    return;
  }
  std::cout << "Got radar data." << std::endl;
  PrintRadarMetadata(burst);
  PrintRadarData(burst);
}

int main(int argc, char *argv[]) {  
	//Priority
	realtime_priority(50, SCHED_RR, -10);

	//Socket part
	int sock = 0, valread;
	struct sockaddr_in serv_addr;
	char *hello = "Hello from soli client";
	char *ready = "soli ready";
	char *complete = "Complete";
	char buffer[1024] = {0};
	if ((sock = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
	{
	printf("\n Socket creation error \n");
	return -1;
	}

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(PORT);
	serv_addr.sin_addr.s_addr = INADDR_ANY;

	// Convert IPv4 and IPv6 addresses from text to binary form
	if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0) 
	{
	printf("\nInvalid address/ Address not supported \n");
	return -1;
	}	
	if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
	{
	printf("\nConnection Failed \n");
	return -1;
	}
	send(sock , hello , strlen(hello) , 0 );
	printf("Hello message sent\n");
	//valread = read( sock , buffer, 1024);
	//printf("%s\n",buffer );
    
    	//Soli part
	file.open("../data/radar.json");
	file << "{\"bursts\":[";

	Soli soli;
	soli.RegisterBurstCallback(&BurstDataCallback);

	// Initialize the radar sensor with the default configuration.
	soli.Init();

	// Customize the chirps per burst and samples per chirp.
	soli.SetConfig("chirps_per_burst", 16);
	soli.SetConfig("samples_per_chirp", 128);
	soli.SetConfig("lower_freq", 59000);
	soli.SetConfig("upper_freq", 61000);
	soli.SetConfig("chirp_rate", 2000);
	// Reset the sensor as we have changed the config.
	soli.Reset();

	//Wait 5 seconds to warm up (also gives the camera time to warm up)
	std::this_thread::sleep_for(std::chrono::seconds(5));
	
	// Start streaming radar data, this will trigger the burst callbacks.
	soli.Start();
	
	//Signal camera to start acquiring frames
	send(sock , ready , strlen(ready) , 0 );
	printf("Ready message sent\n");
	
	// Run for a few seconds, then stop.
	std::this_thread::sleep_for(std::chrono::seconds(60));
	soli.Stop();
	
	send(sock , complete , strlen(complete) , 0 );
	printf("Complete message sent\n");
	
	file.seekp(-1, std::ios_base::cur);
	file << "]}";

	return 0;
	}
