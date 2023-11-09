#ifndef MUSTANG_CLIENT_TYPES_H_
#define MUSTANG_CLIENT_TYPES_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// SPI interface API.
typedef int (*pIfSPIReadCB)(const uint8_t* writeBuf, uint32_t writeBufSize,
                            uint8_t* readBuf, uint32_t readBufSize);
typedef int (*pIfSPIWriteCB)(const uint8_t* writeBuf, uint32_t writeBufSize);

// GPIO interface API.
typedef bool (*pIfGpioGet)(bool* isHigh);
typedef bool (*pIfGpioSet)(bool isHigh);

// Soli event types.
typedef struct SoliPresenceData_ {
  int32_t detected;
  float distance;
  float velocity;
  float likelihood;
} SoliPresenceData_t;

typedef struct SoliReachData_ {
  int32_t detected;
  float distance;
  float velocity;
  float likelihood;
  float azimuth;
  float elevation;
} SoliReachData_t;

typedef struct SoliOmniswipeData_ {
  int32_t detected;
  float likelihood;
  float distance;
} SoliOmniswipeData_t;

typedef struct SoliFlickData_ {
  int32_t detected;
  int32_t direction;
  float likelihood;
  float distance;
} SoliFlickData_t;

typedef struct SoliTapData_ {
  int32_t detected;
} SoliTapData_t;

typedef struct SoliRawMeta_ {
  uint16_t sensor_frame_number;
  uint16_t sensor_adc_temp;
  uint16_t dsp_frame_number;
  uint32_t dsp_ts_ms;
  uint16_t samples_per_chirp;
} SoliRawMeta_t;

// Direction of gestures. Included in flick event data.
typedef enum {
  SOLI_DIRECTION_UNKNOWN_DIRECTION = 0,
  SOLI_DIRECTION_EAST = 1,
  SOLI_DIRECTION_NORTH_EAST = 2,
  SOLI_DIRECTION_NORTH = 3,
  SOLI_DIRECTION_NORTH_WEST = 4,
  SOLI_DIRECTION_WEST = 5,
  SOLI_DIRECTION_SOUTH_WEST = 6,
  SOLI_DIRECTION_SOUTH = 7,
  SOLI_DIRECTION_SOUTH_EAST = 8
} MustangFlickDirection;

// Soli event callback functions typs.
typedef void (*pSoliPresenceCB)(SoliPresenceData_t data);
typedef void (*pSoliReachCB)(SoliReachData_t data);
typedef void (*pSoliOmniswipeCB)(SoliOmniswipeData_t data);
typedef void (*pSoliFlickCB)(SoliFlickData_t data);
typedef void (*pSoliTapCB)(SoliTapData_t data);
typedef void (*pSoliRawCB)(SoliRawMeta_t meta, uint8_t* buf, int32_t size);
typedef void (*pSoliRangeDopplerCB)(uint8_t* buf, int32_t size);

typedef struct {
  uint32_t id;
  const char* name;
  const char* desc;
} SensorCommandInfo;

typedef struct {
  uint32_t id;
  const char* name;
  const char* desc;
} SensorParamInfo;

#ifdef __cplusplus
}
#endif

#endif // MUSTANG_CLIENT_TYPES_H_

