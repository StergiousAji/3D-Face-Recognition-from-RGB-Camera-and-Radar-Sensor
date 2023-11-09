#ifndef MUSTANG_CLIENT_H_
#define MUSTANG_CLIENT_H_

#include "MustangClientTypes.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Performs a complete MustangClient setup using
 *        embedded interface implementation for the current
 *        platform. The setup sequence is:
 *          IfSpiOpen
 *          IfGpioOpen
 *          McInit
 *          McRegisterSpiInterface(IfSpiRead, IfSpiWrite)
 *          McRegisterGpioResetN(IfGpioGetResetNLevel, IfGpioSetResetNLevel)
 *          McConnect
 * @return true if the setup completed successfully.
 *         false if an error occurred.
 */
bool McSetup(void);

/**
 * @brief Performs a complete MustangClient teardown using
 *        embedded interface implementation for the current
 *        platform. The teardown sequence is:
 *          IfSpiClose
 *          IfGpioClose
 *          McDeinit
 * @return true if the teardown completed successfully.
 *         false if an error occurred.
 */
bool McTeardown(void);

/**
 * @brief Initialize the MustangClient library.
 *        This function must be called the first.
 *
 * @return true if the library has been initialized successfully.
 *         false if an error occurred.
 */
bool McInit(void);

/**
 * @brief Deinitialize the MustangClient library.
 *        This function must be called the last.
 *
 * @return true if the library has been deinitialized successfully.
 *         false if an error occurred.
 */
bool McDeinit(void);

/**
 * @brief Register SPI interface to the MustangClient library.
 *        read and write function pointers will be used to communicate
 *        with connected Mustang board.
 *
 * @param spiRead implementation of the read function for a SPI interface.
 * @param spiWrite implementation of the write function for a SPI interface.
 *
 * @return true if read and write functions have been successfully registered.
 *         false if an error occurred.
 */
bool McRegisterSpiInterface(pIfSPIReadCB spiRead,
                            pIfSPIWriteCB spiWrite);

/**
 * @brief Register the GPIO interface for RESETN pin line.
 *
 * @param getResetN implementation of getting the level of the RESETN pin.
 * @param setResetN implementation of setting the level for the RESETN pin.
 *
 * @return true if the GPIO interface has been successfully registered.
 *         false if an error occurred.
 */
bool McRegisterGpioResetN(pIfGpioGet getResetN, pIfGpioSet setResetN);

/**
 * @brief Register the GPIO interface for BOOT_SEL pin line.
 *
 * @param getResetN implementation of getting the level of the BOOT_SEL pin.
 * @param setResetN implementation of setting the level for the BOOT_SEL pin.
 *
 * @return true if the GPIO interface has been successfully registered.
 *         false if an error occurred.
 */
bool McRegisterGpioBootSel(pIfGpioGet getBootSel, pIfGpioSet setBootSel);

/**
 * @brief Once the SPI and GPIOs for RESETN and BOOT_SEL registered, use
 *        this function to establish connection with the Mustang.
 *        Current function also performs the initial check of the Mustang's
 *        state and, if needed, sets the GPIO pins into the proper state.
 *
 * @return true if connection has been established.
 *         false if an error occurred.
 */
bool McConnect(void);

/**
 * @brief Start the data flow from the Mustang board to the MustangClient library.
 *        Should be called when connection is already established and application
 *        is ready to accept Soli gesture events or raw data.
 *
 * @return true if the data flow has been started successfully.
 *         false if an error occurred.
 */
bool McStart(void);

/**
 * @brief Stop the data flow from the Mustang board to the MustangClient library.
 *
 * @return true if the data flow has been stopped successfully.
 *         false if an error occurred.
 */
bool McStop(void);

/**
 * @brief Enable streaming Soli gestures. This option is enabled by default.
 *        The events will be dilivered via registered callbacks.
 *        Note: only streaming Soli gestures, raw data or range doppler
 *        can be enabled at a time. Enablding one type of data will
 *        automatically disable streaming of other data types.
 *
 * @return true if Soli gesture stream enabled successfully.
 *         false if an error occurred.
 */
bool McEnableSoliGestures(void);

/**
 * @brief Register the callback for Soli Presence event.
 *
 * @param callback the function that will be invoked on
 *        Presence event.
 *
 * @return true if the callback has been registered successfully.
 *         false if an error occurred.
 */
bool McSetOnSoliPresenceCB(pSoliPresenceCB callback);

/**
 * @brief Register the callback for Soli Reach event.
 *
 * @param callback the function that will be invoked on
 *        Reach event.
 *
 * @return true if the callback has been registered successfully.
 *         false if an error occurred.
 */
bool McSetOnSoliReachCB(pSoliReachCB callback);

/**
 * @brief Register the callback for Soli Omniswipe event.
 *
 * @param callback the function that will be invoked on
 *        Omniswipe event.
 *
 * @return true if the callback has been registered successfully.
 *         false if an error occurred.
 */
bool McSetOnSoliOmniswipeCB(pSoliOmniswipeCB callback);

/**
 * @brief Register the callback for Soli Flick event.
 *
 * @param callback the function that will be invoked on
 *        Flick event.
 *
 * @return true if the callback has been registered successfully.
 *         false if an error occurred.
 */
bool McSetOnSoliFlickCB(pSoliFlickCB callback);

/**
 * @brief Register the callback for Soli Tap event.
 *
 * @param callback the function that will be invoked on
 *        Tap event.
 *
 * @return true if the callback has been registered successfully.
 *         false if an error occurred.
 */
bool McSetOnSoliTapCB(pSoliTapCB callback);

/**
 * @brief Enable streaming Soli raw data. Data will be delivered via
 *        a registered callback pSoliRawCB.
 *        Note: only streaming Soli gestures, raw data or range doppler
 *        can be enabled at a time. Enablding one type of data will
 *        automatically disable streaming of other data types.
 *
 * @return true if Soli raw data stream enabled successfully.
 *         false if an error occurred.
 */
bool McEnableSoliRaw(void);

/**
 * @brief Register the callback for Soli raw data.
 *
 * @param callback the function that will be invoked on
 *        every new Soli raw data package available.
 *
 * @return true if the callback has been registered successfully.
 *         false if an error occurred.
 */
bool McSetOnSoliRawCB(pSoliRawCB callback);

/**
 * @brief Enable streaming Soli range doppler data. Data will be delivered via
 *        a registered callback pSoliRangeDopplerCB.
 *        Note: only streaming Soli gestures, raw data or range doppler
 *        can be enabled at a time. Enablding one type of data will
 *        automatically disable streaming of other data types.
 *
 * @return true if raw Soli range doppler stream enabled successfully.
 *         false if an error occurred.
 */
bool McEnableSoliRangeDoppler(void);

/**
 * @brief Register the callback for Soli range doppler data.
 *
 * @param callback the function that will be invoked on
 *        every new Soli range doppler data package available.
 *
 * @return true if the callback has been registered successfully.
 *         false if an error occurred.
 */
bool McSetOnSoliRangeDopplerCB(pSoliRangeDopplerCB callback);

/**
 * @brief Invert GPIO logic. It affects the way MC library
 *        handles BOOT_SEL and RESETN lines connected to
 *        Mustang (DSP Chelsea) board.
 * @note  Current function should be called only after McInit.
 *
 * @param enable flag to enable GPIO logic inversion.
 *
 * @return true if GPIO logic inversion is set.
 */
bool McInvertGpioLogic(bool enable);

/**
 * @brief Get the current library version.
 *
 * @param major pointer where major version number will be set
 * @param minor pointer where minor verison number will be set.
 * @param patch pointer where patch version number will be set.
 * @param branch pointer where the branch name will be set.
 * @param commit pointer where the commit ID will be set.
 *
 * @return true if the version number were successfully retrieved.
 */
bool McGetVersion(int* major, int* minor, int* patch,
    char** branch, char** commit);

#ifdef __cplusplus
}
#endif

#endif // MUSTANG_CLIENT_H_

