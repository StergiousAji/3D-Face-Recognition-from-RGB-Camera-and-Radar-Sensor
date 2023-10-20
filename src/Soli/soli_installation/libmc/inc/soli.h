#ifndef _SOLI_H_
#define _SOLI_H_

#include <cstdint>
#include <string>
#include <functional>
#include <vector>
#include <memory>

#include <MustangClientTypes.h>
#include <soli_radar_data.h>

// Forward declaration.
class SoliDataStreamer;

// A simple class for interfacing with the Soli sensor.
//
class Soli {
  public:
    Soli();
    ~Soli();
    
    // Initializes the sensor.
    //
    // Called automatically the first time Start is called if no prior call
    // to Init has already been made.
    //
    // Returns true if successful, false otherwise.
    bool Init();

    // Starts data streaming from the sensor.
    //
    // Returns true if successful, false otherwise.
    bool Start();

    // Stops data streaming from the sensor.
    // Returns true if successful, false otherwise.
    bool Stop();
    
    // Resets the radar. This should be called anytime a
    // config change has been made.
    bool Reset();

    // Sets config value to sensor.
    //
    // Returns true if successful, false otherwise.
    bool SetConfig(const std::string& key, uint32_t value);

    // Registers the callback function that should be triggered on each new incoming radar burst.
    void RegisterBurstCallback(std::function<void(const SoliRadarBurst&)> callback);
  private:
    std::unique_ptr<SoliDataStreamer> soli_;
};

#endif

