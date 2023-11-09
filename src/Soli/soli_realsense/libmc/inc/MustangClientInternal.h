#ifndef MUSTANG_CLIENT_INTERNAL_H_
#define MUSTANG_CLIENT_INTERNAL_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Program the Mustang's flash. Usually this is needed when a Mustang
 *        board is received from the factory with the empty flash memory.
 *        Function call is available when MustangClient is connected to the
 *        Mustang board.
 *
 * @param flashBootSectorImg file path to the flash bootsector image.
 * @param bootstrapImg file path to the bootstrap image.
 * @param firmwareImg file path to the firmware image.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McFwProgram(const char* flashBootSectorImg,
                 const char* bootstrapImg,
                 const char* firmwareImg);

/**
 * @brief Upload only the firmware image to the RAM. Useful you want to try
 *        a FW image without changing the image in the flash.
 *        Function call is available when MustangClient is connected to the
 *        Mustang board.
 *        This is the stage 1 of the McFwProgram process.
 *
 * @param firmwareImg file path to the firmware image.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McFwUpgradeRam(const char* firmwareImg);

/**
 * @brief Program the Mustang's flash only with the provided firmware image.
 *        This will replace the flash memory content with the content form
 *        the file as is.
 *        This is the stage 2 of the McFwProgram process.
 *
 * @param flashImg file path to the flash image.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McFwProgramFlash(const char* flashImg);

/**
 * @brief Upgrade the available partition on the flash
 *        with the provided firmware image.
 *        This is the stage 3 of the McFwProgram process.
 *
 * @param firmwareImg file path to the firmware image.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McFwUpgradeFlash(const char* firmwareImg);

/**
 * @brief Erase the Mustang's flash memory completely.
 *        This is meant to be used when the board needs to be turned to the
 *        state as it was received from the factory.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McFwEraseFlash(void);

// Callback signature for raw tunnal data streaming.
typedef void(*pStreamDataCB)(uint32_t* data, int words);

/**
 * @brief Register the callback for streaming raw tunnel data.
 *
 * @param streamHandler a callback function that will be invoked
 *        when a new data available in the DSP tunnel.
 *
 * @return true if the callback function has been successfully registered.
 *         false if an error occurred.
 */
bool McSetStreamCB(pStreamDataCB streamHandler);

/**
 * @brief Turn on the Mustang board.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McTurnOn();

/**
 * @brief Turn off the Mustang board.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McTurnOff();

/**
 * @brief Set the system mode to boot mode.
 *        When Mustang board is turned on in boot mode, it can
 *        perform firmware upgrade operations.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McSetBootMode();

/**
 * @brief Set the system mode to application mode.
 *        When Mustang board is turned on in application mode, it
 *        will try to load the firmware image from the flash. If
 *        flash does not have a valid image, it will switch to
 *        boot mode.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McSetAppMode();

void McPrintDebugLogs();

// This is a copy of tunnel_frame_t except 64bit timestamp is
// split into 2 halves with 32 bits each.
// Since on 64 bit systems, 4 bytes padding will be added after
// crc field and will mess the protocol integrity.
typedef struct tunnel_header_s {
  uint32_t  magicNumber;
  uint16_t  id;
  uint16_t  srcEpId;
  uint32_t  crc;
  uint32_t  timeStamp1;
  uint32_t  timeStamp2;
  uint32_t  seqNo;
  uint16_t  frameSizeInBytes;
  uint8_t   encoding;
  uint8_t   sampleRate;
  uint32_t  data[0];
} tunnel_header_t;

#ifdef __cplusplus
}
#endif

#endif // MUSTANG_CLIENT_INTERNAL_H_

