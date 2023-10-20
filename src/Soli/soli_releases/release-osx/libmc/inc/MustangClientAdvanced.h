#ifndef MUSTANG_CLIENT_ADVANCED_H_
#define MUSTANG_CLIENT_ADVANCED_H_

#include "MustangClientTypes.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get the Mustang's system mode.
 *        Mode 1: Boot mode. Ready to accept FW image.
 *        Mode 2: Applicatoin mode. Running loaded FW image.
 *
 * @param mode pointer to where the system mode will be written.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McGetBootMode(uint32_t* mode);

/**
 * @brief Get the Mustang's DSP device ID.
 *
 * @param deviceId pointer to where the device ID will be written.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McReadDeviceId(uint32_t* deviceId);

/**
 * @brief Get the Mustang's Rome framework version number.
 *
 * @param versionNum pointer to where the version number will be written.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McReadRomeVerNum(uint32_t* versionNum);

/**
 * @brief Get the Mustang's Rome framework version string.
 *
 * @param versionStr pointer to where the version string will be written.
 * @param size the maximum bytes versionStr buffer can hold.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McReadRomeVerStr(char* versionStr, int size);

/**
 * @brief Get the Mustang's Application version number.
 *
 * @param versionNum pointer to where the version number will be written.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McReadAppVerNum(uint32_t* versionNum);

/**
 * @brief Get the Mustang's Application version string.
 *
 * @param versionStr pointer to where the version string will be written.
 * @param size the maximum bytes versionStr buffer can hold.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McReadAppVerStr(char* versionStr, int size);

/**
 * @brief Get the Soli sensor commands' information.
 *
 * @param commands the pointer which will be set to the list with
 *        sensor commands' information.
 * @param count the pointer to the value which will be set to the
 *        amount of sensor commands available.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McGetSensorCommandsInfo(SensorCommandInfo** commands, int* count);

/**
 * @brief Get the Soli sensor parameters' information.
 *
 * @param params the pointer which will be set to the list with
 *        sensor parameters' information.
 * @param count the pointer to the value which will be set to the
 *        amount of sensor parameters available.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McGetSensorParamsInfo(SensorParamInfo** params, int* count);

/**
 * @brief Get the Soli sensor's parameter. Please refer to the documentation
 *        for available param IDs and its meaning.
 *
 * @param paramId parameter ID value of which you want to get.
 * @param paramVal pointer to where parameter value will be written.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McGetSensorParam(uint32_t paramId, uint32_t* paramVal);

/**
 * @brief Set the Soli sensor's parameter. Please refer to the documentation
 *        for available param IDs and its meaning.
 *
 * @param paramId parameter ID value of which you want to set.
 * @param paramVal parameter value you want to set.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McSetSensorParam(uint32_t paramId, uint32_t paramVal);

/**
 * @brief Get the Soli plugin's parameter. Please refer to the documentation
 *        for available param IDs and its meaning.
 *
 * @param paramId parameter ID value of which you want to get.
 * @param paramVal pointer to where parameter value will be written.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McGetPluginParam(uint32_t paramId, uint32_t* paramVal);

/**
 * @brief Set the Soli plugin's parameter. Please refer to the documentation
 *        for available param IDs and its meaning.
 *
 * @param paramId parameter ID value of which you want to set.
 * @param paramVal parameter value you want to set.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McSetPluginParam(uint32_t paramId, uint32_t paramVal);

/**
 * @brief Upgrade Mustang's firmware image.
 *
 * @param bootstrapImg the file path to a bootstrap image.
 * @param firmwareImg the file path to a new firmware image to upgrade to.
 *
 * @return true if operation finished successfully.
 *         false if an error occurred.
 */
bool McFwUpgrade(const char* bootstrapImg, const char* firmwareImg);

#ifdef __cplusplus
}
#endif

#endif // MUSTANG_CLIENT_ADVANCED_H_

