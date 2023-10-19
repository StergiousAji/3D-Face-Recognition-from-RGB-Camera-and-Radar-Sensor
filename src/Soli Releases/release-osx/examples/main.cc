// A simple Soli example demonstrating on how to get radar data from the sensor.

#include <chrono>
#include <iostream>
#include <thread>

#include "soli.h"

void PrintRadarMetadata(const SoliRadarBurst& burst) {
  if (!burst.is_valid) return;
  const int num_chirps_per_burst = burst.chirp_data.size();
  const int num_rx_channels = burst.chirp_data[0].channel_data.size();
  const int num_samples_per_chirp = burst.chirp_data[0].channel_data[0].size();
  std::cout << "-burst id: " << burst.burst_id << std::endl;
  std::cout << "-DSP timestamp: " << burst.timestamp_ms << std::endl;
  std::cout << "-num chirps per burst: " << num_chirps_per_burst << std::endl;
  std::cout << "-num samples per chirp: " << num_samples_per_chirp << std::endl;
  std::cout << "-num rx channels: " << num_rx_channels << std::endl;
}

void PrintRadarData(const SoliRadarBurst& burst) {
  if (!burst.is_valid) return;
  std::cout << "-radar data:" << std::endl;
  for (const SoliRadarChirp& chirp_data : burst.chirp_data) {
    for (size_t rx_channel = 0; rx_channel < chirp_data.channel_data.size(); rx_channel++) {
      for (const float value : chirp_data.channel_data[rx_channel]) {
        std::cout << value << " ";
      }
    }
  }
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
  Soli soli;
  soli.RegisterBurstCallback(&BurstDataCallback);

  // Initialize the radar sensor with the default configuration.
  soli.Init();

  // Customize the chirps per burst and samples per chirp.
  soli.SetConfig("chirps_per_burst", 16);
  soli.SetConfig("samples_per_chirp", 64);
  // Reset the sensor as we have changed the config.
  soli.Reset();

  // Start streaming radar data, this will trigger the burst callbacks.
  soli.Start();

  // Run for a few seconds, then stop.
  std::this_thread::sleep_for(std::chrono::seconds(5));
  soli.Stop();

  return 0;
}
