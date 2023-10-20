#ifndef _SOLI_RADAR_DATA_H_
#define _SOLI_RADAR_DATA_H_

#include <cstdint>
#include <vector>

// Represents the received radar chirp for each
// radar receiver channel.
typedef struct SoliRadarChirp {
  // The timestamp of the chirp in microseconds.
  int64_t timestamp_usec;
  // The chrip data for each channel. The outer vector
  // represents each channel, the inner vector represents
  // the received chirp data.
  std::vector<std::vector<float>> channel_data;
} SoliRadarChirp;

typedef struct SoliRadarBurst {
  // True if the radar data is deemed to be valid, false otherwise.
  bool is_valid;
  // The monotonically increasing index for a given burst.
  int32_t burst_id;
  // The timestamp of the burst in milliseconds.
  int64_t timestamp_ms;
  // The vector of chirps within the radar burst.
  std::vector<SoliRadarChirp> chirp_data;
} SoliRadarBurst;

#endif

