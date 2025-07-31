#ifndef E1701_H
#define E1701_H

#include <stdint.h>

// *************************************************************************************************
// *** for any new developments please use the E170X-defines only, the E1701D are deprecated     ***
// *** and may become obsolete with a new hardware variant and the related programming interface ***
// *************************************************************************************************

#ifndef E170X_DLL_VERSION
// this is necessary to avoid collisions between old libe1701 and new libe170x; when both are used in same application,
// E170X_-functions are disabled for libe1701 and the old-style E1701_-functions need to be used

#define E170X_OK                        0 // operation could be finished successfully
#define E170X_ERROR_INVALID_CARD      101 // wrong/illegal card number specified
#define E170X_ERROR_NO_CONNECTION     102 // could not connect to card
#define E170X_ERROR_NO_MEMORY         103 // not enough memory available
#define E170X_ERROR_UNKNOWN_FW        104 // unknown/incompatible firmware version
#define E170X_ERROR                   105 // unknown/unspecified error
#define E170X_ERROR_TRANSMISSION      106 // transmission of data failed
#define E170X_ERROR_FILEOPEN          107 // opening a file failed
#define E170X_ERROR_FILEWRITE         108 // writing data to a file failed
#define E170X_ERROR_BORD_NA           109 // a base- or extension board that would be required for a function is not available
#define E170X_ERROR_INVALID_DATA      110 // a function was called with invalid data or by using an operation mode where this function is not used/allowed
#define E170X_ERROR_UNKNOWN_BOARD     111 // trying to access a board that is not a scanner controller
#define E170X_ERROR_FILENAME          112 // there is an error with the given filename (too long, too many subdirectories, illegal characters,...)
#define E170X_ERROR_NOT_SUPPORTED     113 // the requested feature is not supported by the current firmware version
#define E170X_ERROR_NO_DATA_AVAILABLE 114 // tried to receive some data but there are none avilable yet
#define E170X_ERROR_FILEREAD          115 // reading data out of a file failed
#define E170X_ERROR_STILL_IN_PROGRESS 116 // the controller is still busy, e.g. with some runnign stand-alone projects

#define E170X_MAX_HEAD_NUM         16 // maximum number of cards that can be controlled

#define E170X_LASER_FPK         0x90000000 // for backwards compatibility set two bits here
#define E170X_LASER_FREQ_ON1    0x40000000 // switch on-frequency/q-switch immediately
#define E170X_LASER_FREQ_ON2    0x20000000 // switch on-frequency/q-switch after yag3QTime; in mode YAG 2 it is equal to FPK time
#define E170X_LASER_FREQ_OFF    0x08000000 // output a stand-by frequency during jumps
#define E170X_LASER_FREQ_DUAL   0x04000000 // output a second frequency at LaserB permanently

#define E170X_LASERMODE_CO2     (1|E170X_LASER_FREQ_ON1|E170X_LASER_FREQ_OFF)
#define E170X_LASERMODE_YAG1    (2|E170X_LASER_FREQ_ON1|E170X_LASER_FREQ_OFF|E170X_LASER_FPK)
#define E170X_LASERMODE_YAG2    (3|E170X_LASER_FREQ_ON2|E170X_LASER_FREQ_OFF|E170X_LASER_FPK)
#define E170X_LASERMODE_YAG3    (4|E170X_LASER_FREQ_ON2|E170X_LASER_FREQ_OFF|E170X_LASER_FPK)
#define E170X_LASERMODE_CRF     (5|E170X_LASER_FREQ_ON1) // laser mode with continuously running frequency
#define E170X_LASERMODE_DFREQ   (6|E170X_LASER_FREQ_ON1|E170X_LASER_FREQ_OFF|E170X_LASER_FREQ_DUAL) // dual frequency laser mode which emits a second frequency at LaserB
#define E170X_LASERMODE_unused  (7)
#define E170X_LASERMODE_MOPA    (8|E170X_LASER_FREQ_ON1|E170X_LASER_FREQ_OFF)

#define E170X_CSTATE_MARKING              0x0000001
#define E170X_CSTATE_START_PRESSED        0x0000002
#define E170X_CSTATE_STOP_PRESSED         0x0000004
#define E170X_CSTATE_WAS_START_PRESSED    0x0000008
#define E170X_CSTATE_WAS_STOP_PRESSED     0x0000010
#define E170X_CSTATE_ERROR                0x0000020
#define E170X_CSTATE_WAS_EXTTRIGGER       0x0000040 // internal use only, do not check this flag
#define E170X_CSTATE_PROCESSING           0x0000080
#define E170X_CSTATE_EMITTING             0x0000100
#define E170X_CSTATE_FILE_WRITE_ERROR     0x0000200
#define E170X_CSTATE_WAIT_EXTTRIGGER      0x0000400
#define E170X_CSTATE_WAS_SILENTTRIGGER    0x0000800 // internal use only, do not check this flag
#define E170X_CSTATE_FILEMODE_ACTIVE      0x0001000 // internal use only, do not check this flag
#define E170X_CSTATE_HALTED               0x0002000
#define E170X_CSTATE_WRITING_DATA         0x0004000
#define E170X_CSTATE_WRITING_DATA_ERROR   0x0008000
#define E170X_CSTATE_WAS_MOTION_STARTED   0x0010000
#define E170X_CSTATE_WAS_MOTION_STOPPED   0x0020000
#define E170X_CSTATE_IS_REFERENCING       0x0040000 // not used by E1701A/D but by E1701C, defined here due to common firmware use
#define E170X_CSTATE_unused2              0x0080000
#define E170X_CSTATE_WAIT_INPUT           0x0100000
#define E170X_CSTATE_SAC_READY            0x0200000 // in stand alone-mode only: similar to DOut0, signals "ready for marking"
#define E170X_CSTATE_SAC_MARKING          0x0400000 // in stand alone-mode only: similar to DOut1, signals "marking active"
#define E170X_CSTATE_SAC_CTLXY            0x0800000 // in stand alone-mode only: command ctlxy was sent and laser is on until ExtStart is released
#define E170X_CSTATE_WAIT_EXTSIGNAL       0x1000000
//#define E170X_CSTATE_DEBUG              0x80000000

#define E170X_MAX_CORRECTION_TABLES 16

#define E170X_FILEMODE_OFF   0xFFFFFFFF
#define E170X_FILEMODE_LOCAL 0
#define E170X_FILEMODE_SEND  1

#define E170X_BSTATE_XY2_100_BB        0x0001
#define E170X_BSTATE_ILDA_BB           0x0002
//#define E170X_BSTATE_CNC_BB            0x0004
#define E170X_BSTATE_LP8_EB            0x0100
#define E170X_BSTATE_DIGI_EB           0x0200
#define E170X_BSTATE_LY001_BB          0x0400

#ifndef E170X_BSTATE_BB_MASK
 #define E170X_BSTATE_BB_MASK          (E170X_BSTATE_XY2_100_BB|E170X_BSTATE_ILDA_BB|E170X_BSTATE_LY001_BB)
#endif

#define E170X_TUNE_EXTTRIG_DIGIIN7     0x00000001
#define E170X_TUNE_2D_MOTF             0x00000002
#define E170X_TUNE_SAVE_SERIAL_STATES  0x00000004 // when this option is set the current state of serial numbers is stored during marking and does not get los on power cycle
#define E170X_TUNE_INVERT_LASERGATE    0x00000008
#define E170X_TUNE_INVERT_LASERA       0x00000010
#define E170X_TUNE_INVERT_LASERB       0x00000020
#define E170X_TUNE_LASERA_GPO          0x00000040
#define E170X_TUNE_LASERB_GPO          0x00000080
// #define E170X_TUNE_10V_ANALOGUE_XYZ    0x00000100 E1701A parameter is deprecated
#define E170X_TUNE_USE_A1_AS_Z         0x00000200
#define E170X_TUNE_STUPI2D_XY2         0x00000400
//#define E170X_TUNE_FAST_MATRIX         0x00000800 always turned on now
#define E170X_TUNE_XY2_18BIT           0x00001000
#define E170X_TUNE_XY3_20BIT           0x00002000
#define E170X_TUNE_DISABLE_TEST        0x00004000
#define E170X_TUNE_INVERT_MIP          0x00008000
#define E170X_TUNE_INVERT_WET          0x00010000
#define E170X_TUNE_EXTTRIG_DIGIIN6     0x00020000
//#define E170X_TUNE_EXACT_CORRECTION    0x040000 // enabled by default since firmware version 40
#define E170X_TUNE_INVERT_EXTSTOP      0x00080000 // invert ExtStop logic, if set, it needs to be at HIGH in order to not to be stopped
//#define E170X_TUNExxx                  0x100000
#define E170X_TUNE_HALT_WITH_EXTSTART  0x00200000
#define E170X_TUNE_INVERT_LP8          0x00400000
#define E170X_TUNE_INVERT_MO           0x00800000
#define E170X_TUNE_INVERT_EXTSTART     0x01000000 // invert ExtStart logic including E170X_TUNE_HALT_WITH_EXTSTART functionality
#define E170X_TUNE_QUICK_STARTUP       0x02000000
#define E170X_TUNE_FORCE_TO_ZERO       0x04000000 // force scanhead to zero position on start
//#define E170X_TUNE_EXCHANGE_XY         0x40000000 unused
#define E170X_TUNE_DONOTUSE            0x80000000
#define E170X_TUNE_SCANNERMODE_MASK    (E170X_TUNE_STUPI2D_XY2|E170X_TUNE_XY2_18BIT|E170X_TUNE_XY3_20BIT)

#define E170X_COMMAND_FLAG_STREAM      0x0001 // command has to be enqueued in stream
#define E170X_COMMAND_FLAG_DIRECT      0x0002 // command has to be executed directly and immediately
#define E170X_COMMAND_FLAG_PASSIVE     0x0004 // do not send a request to the hardware but use data which are already buffered
#define E170X_COMMAND_FLAG_SILENT      0x0008 // do not let this function cause a state-change but let it operate in "background" silently
#define E170X_COMMAND_FLAG_HIGH_LEVEL  0x0010 // use a logical high-level, when not set, work with logical low-level

#define E170X_COMMAND_FLAG_WRITE_MASK             0x0F00
#define E170X_COMMAND_FLAG_WRITE_LP8MO            0x0100
#define E170X_COMMAND_FLAG_WRITE_LP8LATCH         0x0200
#define E170X_COMMAND_FLAG_WRITE_LASERA_GPO       0x0300
#define E170X_COMMAND_FLAG_WRITE_LASERB_GPO       0x0400
#define E170X_COMMAND_FLAG_WRITE_LASERGATE        0x0500
//#define E170X_COMMAND_FLAG_WRITE_SLICETHICKNESS   0x0600
#define E170X_COMMAND_FLAG_WRITE_SPOTSIZE         0x0700
#define E170X_COMMAND_FLAG_WRITE_LASERA_GPO_PULSE 0x0800

#define E170X_COMMAND_FLAG_DYNDATA_MARK_FONTENTRY 0x0100

#define E170X_COMMAND_FLAG_MOTF_WAIT_INCS         0x0100
#define E170X_COMMAND_FLAG_MOTF_WAIT_BITS         0x0200

#define E170X_COMMAND_FLAG_XYCORR_FLIPXY       0x0100
#define E170X_COMMAND_FLAG_XYCORR_MIRRORX      0x0200
#define E170X_COMMAND_FLAG_XYCORR_MIRRORY      0x0400
#define E170X_COMMAND_FLAG_ZCORR_MIRRORZ       0x0800

#define E170X_COMMAND_FLAG_SCANNER_VAR_POLYDELAY  0x0100

#define E170X_COMMAND_FLAG_ANA_AOUT0     0x00000100 // analogue output
#define E170X_COMMAND_FLAG_ANA_MASK      (E170X_COMMAND_FLAG_ANA_AOUT0)
                                                    // mask when used in context of analogue read/write functions

#define E170X_PIXELMODE_NO_JUMPS           0x0001 // do not perform jumps also in case power is 0%; when this flag is set, power threshold is ignored
#define E170X_PIXELMODE_JUMP_N_SHOOT       0x0002 // no continuous movement, jump to the marking position and shoot there for laseroff minus laseron time
#define E170X_PIXELMODE_HW_POWER_CONTROL   0x0004 // power control is done by hardware internally, this is currently supported for E170X_LASERMODE_MOPA/E170X_LASERMODE_CO2/E170X_LASERMODE_YAGx only
#define E170X_PIXELMODE_GATE_POWER_CONTROL 0x0008 // special mode suitable for b/w bitmaps, laser gate is switched on/off depending on power >= or < 50%
#define E170X_PIXELMODE_JUMP_LEAVE_POWER   0x0010 // do not turn off the power during jumps within a pixel line (not to be set for some lasers with emissions during LaserGate being off)

#define E170X_FREE_SPACE_PRIMARY   0
#define E170X_FREE_SPACE_SECONDARY 1
#define E170X_USED_SPACE_QUEUE     2

struct oapc_bin_struct_dyn_data; //forward declaration in case oapc_libio.h is not used
struct oapc_bin_struct_dyn_data2; //forward declaration in case oapc_libio.h is not used

#ifdef __cplusplus
extern "C"
{
#endif
   typedef int (*E170X_power_callback)(unsigned char n,double power,void *userData); /** type definition for the callback function that has to be provided by client for setting power */
   typedef int (*E170X_progress_callback)(unsigned char n,unsigned int progress,void *userData); /** type definition for the callback function that has to be provided by client for getting progress information */
#ifdef __cplusplus
};
#endif

#ifndef ENV_E1701
// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the E1701_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// E170X_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
//#ifdef ENV_LINUX
 #ifdef E1701_EXPORTS
  #define E170X_API __attribute ((visibility ("default")))
 #else
  #define E170X_API
 #endif
/*#else
 #ifdef E1701_EXPORTS
  #define E170X_API __declspec(dllexport)
 #else
  #define E170X_API __declspec(dllimport)
 #endif
#endif*/

#ifdef __cplusplus
extern "C"
{
#else
   typedef char bool; 
#endif
   // ***** E170x easy interface functions *************************************************************************************
   // base functions
   E170X_API unsigned char E170X_set_connection(const char *address);
   E170X_API void          E170X_set_password(const unsigned char n,const char *ethPwd);
   E170X_API int           E170X_set_filepath(unsigned char n,const char *fname,unsigned int mode);
   E170X_API int           E170X_set_debug_logfile(const unsigned char n,const char *path,const unsigned char flags); // for logging of local function calls, suitable for debugging of own application
   E170X_API int           E170X_write_debug_logfile(const unsigned char n,const char *format,...); // for writing own debug texts into log
   E170X_API void          E170X_close(unsigned char n);
   E170X_API int           E170X_load_correction(unsigned char n, const char* filename,unsigned char tableNum);
   E170X_API int           E170X_switch_correction(unsigned char n,unsigned char tableNum);
   E170X_API int           E170X_set_xy_correction(const unsigned char n,const unsigned int flags,const double gainX, const double gainY, const double rot, const int offsetX, const int offsetY, const double slantX, const double slantY);
   E170X_API int           E170X_set_z_correction2(const unsigned char n,const unsigned int flags,const double gainZ,const int offsetZ,const unsigned int h,const double xy_to_z_ratio);
   E170X_API int           E170X_set_z_correction(const unsigned char n,const unsigned int h,const double xy_to_z_ratio,const int res);
   E170X_API int           E170X_tune(const unsigned char n,const unsigned int tuneFlags);
   E170X_API int           E170X_set_speeds(unsigned char n, double jumpspeed,double markspeed);
   E170X_API int           E170X_set_overspeed(const unsigned char n,const unsigned int flags,const double scannerLag,const double jumpFactor,const double reserved);
   E170X_API int           E170X_set_laser_delays(const unsigned char n,const double ondelay,const double offdelay);
   E170X_API int           E170X_set_laser_mode(const unsigned char n,const unsigned int mode);
   E170X_API int           E170X_set_laser(const unsigned char n,const unsigned int flags,const char on);
   E170X_API int           E170X_set_wobble(const unsigned char n,unsigned int x,unsigned int y,double freq);
   E170X_API int           E170X_set_wobble_fuzz(const unsigned char n,const uint8_t mode,uint32_t x,uint32_t y,double freq,double updateFreq);
   E170X_API int           E170X_set_scanner_delays(const unsigned char n,const unsigned int flags,const double jumpdelay,const double markdelay,const double polydelay);
   E170X_API int           E170X_jump_abs(const unsigned char n,int x,int y,int z);
   E170X_API int           E170X_mark_abs(const unsigned char n,int x,int y,int z);
   E170X_API int           E170X_set_pixelmode(const unsigned char n,const unsigned int mode,const double powerThres,const unsigned int res);
   E170X_API int           E170X_mark_pixelline(const unsigned char n,int x,int y,int z,const int pixWidth,const int pixHeight,const int pixDepth,unsigned int pixNum,const double *pixels,E170X_power_callback power_callback,void *userData);
   E170X_API int           E170X_set_pos(const unsigned char n,const int x,const int y,const int z,const unsigned char laserOn);
   E170X_API int           E170X_get_pos(const unsigned char n,int *x,int *y,int *z);
   E170X_API int           E170X_set_matrix(const unsigned char n,const unsigned int flags,const double m11,const double m12,const double m21,const double m22);
   E170X_API int           E170X_set_trigger_point(const unsigned char n);
   E170X_API int           E170X_set_signal_point(const unsigned char n,const unsigned int flags);
   E170X_API int           E170X_release_trigger_point(const unsigned char n);
   E170X_API int           E170X_set_sync(const unsigned char n,const unsigned int flags,const unsigned int value);
   E170X_API unsigned int  E170X_get_sync(const unsigned char n);
   E170X_API int           E170X_set_extstart(const unsigned char n,const unsigned int flags,const unsigned int receivedCtr,const unsigned int expectedCtr);
   E170X_API int           E170X_get_extstart(const unsigned char n,unsigned int *receivedCtr,unsigned int *expectedCtr);
   E170X_API int           E170X_execute(const unsigned char n);
   E170X_API int           E170X_stop_execution(const unsigned char n);
   E170X_API int           E170X_halt_execution(const unsigned char n,unsigned char halt);
   E170X_API int           E170X_delay(const unsigned char n,const double delay);
   E170X_API int           E170X_dynamic_data(const unsigned char n,struct oapc_bin_struct_dyn_data *dynData); // deprecated, use E170X_dynamic_data2 instead
   E170X_API int           E170X_dynamic_data2(const unsigned char n,struct oapc_bin_struct_dyn_data2 *dynData);
   E170X_API int           E170X_dynamic_mark(const unsigned char n,const unsigned int flags,const void *value);
   E170X_API int           E170X_loop(const unsigned char n,const unsigned int flags,const unsigned int repeat);
   E170X_API unsigned int  E170X_get_startstop_state(const unsigned char n);
   E170X_API int           E170X_get_card_state(const unsigned char n, unsigned int *state);
   E170X_API unsigned int  E170X_get_card_info(const unsigned char n);
   E170X_API int           E170X_set_laser_timing(const unsigned char n,double frequency,double pulse);
   E170X_API int           E170X_set_laserb(const unsigned char n,const double frequency,const double pulse);
   E170X_API int           E170X_set_standby(const unsigned char n,const double frequency,const double pulse,const bool force);
   E170X_API int           E170X_set_fpk(const unsigned char n,double fpk,double yag3QTime);
   E170X_API int           E170X_get_free_space(const unsigned char n,int buffer);
   E170X_API void          E170X_get_version(const unsigned char n,unsigned short *hwVersion,unsigned short *fwVersion);
   E170X_API int           E170X_get_library_version();
   E170X_API int           E170X_get_serial_number(const unsigned char n,char *serial,const int length);
   E170X_API int           E170X_write(const unsigned char n,unsigned int flags,unsigned int value);

   // LP8 extension board functions
   E170X_API int           E170X_lp8_write(const unsigned char n,const unsigned int flags,const unsigned char value);
   E170X_API int           E170X_lp8_write_latch(unsigned char n,unsigned char on,double delay1,unsigned char value,double delay2,double delay3);
   E170X_API int           E170X_lp8_ana_write(const unsigned char n,const unsigned int flags,const unsigned short a);
   E170X_API int           E170X_lp8_write_mo(const unsigned char n,const unsigned int flags,const unsigned char on);
   E170X_API int           E170X_lp8_write_pilot(const unsigned char n,const unsigned int flags,const unsigned char on);

   // DIGI I/O extension board functions
   E170X_API int           E170X_digi_write(const unsigned char n,unsigned int flags,unsigned int value,unsigned int mask);
   E170X_API int           E170X_digi_pulse(const unsigned char n,const unsigned int flags,const unsigned int value,const unsigned int mask,const unsigned int pulses,const double delayOn,const double delayOff);
   E170X_API int           E170X_digi_read(const unsigned char n,const unsigned int flags,unsigned int *value);
   E170X_API int           E170X_digi_wait(const unsigned char n,unsigned long value,unsigned long mask);
   E170X_API int           E170X_digi_set_motf(const unsigned char n,double motfX,double motfY);
   E170X_API int           E170X_digi_set_motf_sim(const unsigned char n,double motfX,double motfY);
   E170X_API int           E170X_digi_wait_motf(const unsigned char n,const unsigned int flags,const double dist);
   E170X_API int           E170X_digi_set_mip_output(const unsigned char n,unsigned int value,unsigned int flags);
   E170X_API int           E170X_digi_set_wet_output(const unsigned char n,const unsigned int value,const unsigned int flags);

   // Miscellaneous internal-only functions
   E170X_API unsigned int  E170X_send_data(const unsigned char n,const unsigned int flags,const char *sendData,unsigned int length,unsigned int *sentLength,E170X_progress_callback progress_callback, void *userData,int *error);
   E170X_API unsigned int  E170X_recv_data(const unsigned char n,unsigned int flags,char *recvData,unsigned int maxLength);

   // ***** end of E170x easy interface functions ******************************************************************************
#ifdef __cplusplus
};
#endif // __cplusplus
#endif // ENV_E1701


// *** end of for any new developments please use the E170X-defines only ***
#endif //E170X_DLL_VERSION

#define E1701_DLL_VERSION 1

#define E1701_OK                      E170X_OK
#define E1701_ERROR_INVALID_CARD      E170X_ERROR_INVALID_CARD
#define E1701_ERROR_NO_CONNECTION     E170X_ERROR_NO_CONNECTION
#define E1701_ERROR_NO_MEMORY         E170X_ERROR_NO_MEMORY
#define E1701_ERROR_UNKNOWN_FW        E170X_ERROR_UNKNOWN_FW
#define E1701_ERROR                   E170X_ERROR
#define E1701_ERROR_TRANSMISSION      E170X_ERROR_TRANSMISSION
#define E1701_ERROR_FILEOPEN          E170X_ERROR_FILEOPEN
#define E1701_ERROR_FILEWRITE         E170X_ERROR_FILEWRITE
#define E1701_ERROR_BORD_NA           E170X_ERROR_BORD_NA
#define E1701_ERROR_INVALID_DATA      E170X_ERROR_INVALID_DATA
#define E1701_ERROR_UNKNOWN_BOARD     E170X_ERROR_UNKNOWN_BOARD
#define E1701_ERROR_FILENAME          E170X_ERROR_FILENAME
#define E1701_ERROR_NOT_SUPPORTED     E170X_ERROR_NOT_SUPPORTED
#define E1701_ERROR_NO_DATA_AVAILABLE E170X_ERROR_NO_DATA_AVAILABLE

#define E1701_MAX_HEAD_NUM            E170X_MAX_HEAD_NUM

#define E1701_LASER_FPK         E170X_LASER_FPK
#define E1701_LASER_FREQ_ON1    E170X_LASER_FREQ_ON1
#define E1701_LASER_FREQ_ON2    E170X_LASER_FREQ_ON2
#define E1701_LASER_FREQ_OFF    E170X_LASER_FREQ_OFF
#define E1701_LASER_FREQ_DUAL   E170X_LASER_FREQ_DUAL

#define E1701_LASERMODE_CO2     E170X_LASERMODE_CO2
#define E1701_LASERMODE_YAG1    E170X_LASERMODE_YAG1
#define E1701_LASERMODE_YAG2    E170X_LASERMODE_YAG2
#define E1701_LASERMODE_YAG3    E170X_LASERMODE_YAG3
#define E1701_LASERMODE_CRF     E170X_LASERMODE_CRF
#define E1701_LASERMODE_DFREQ   E170X_LASERMODE_DFREQ
#define E1701_LASERMODE_MOPA    E170X_LASERMODE_MOPA

#define E1701_CSTATE_MARKING              E170X_CSTATE_MARKING
#define E1701_CSTATE_START_PRESSED        E170X_CSTATE_START_PRESSED
#define E1701_CSTATE_STOP_PRESSED         E170X_CSTATE_STOP_PRESSED
#define E1701_CSTATE_WAS_START_PRESSED    E170X_CSTATE_WAS_START_PRESSED
#define E1701_CSTATE_WAS_STOP_PRESSED     E170X_CSTATE_WAS_STOP_PRESSED
#define E1701_CSTATE_ERROR                E170X_CSTATE_ERROR
#define E1701_CSTATE_WAS_EXTTRIGGER       E170X_CSTATE_WAS_EXTTRIGGER // internal use only, do not check this flag
#define E1701_CSTATE_PROCESSING           E170X_CSTATE_PROCESSING
#define E1701_CSTATE_EMITTING             E170X_CSTATE_EMITTING
#define E1701_CSTATE_FILE_WRITE_ERROR     E170X_CSTATE_FILE_WRITE_ERROR
#define E1701_CSTATE_WAIT_EXTTRIGGER      E170X_CSTATE_WAIT_EXTTRIGGER
#define E1701_CSTATE_WAS_SILENTTRIGGER    E170X_CSTATE_WAS_SILENTTRIGGER // internal use only, do not check this flag
#define E1701_CSTATE_FILEMODE_ACTIVE      E170X_CSTATE_FILEMODE_ACTIVE // internal use only, do not check this flag
#define E1701_CSTATE_HALTED               E170X_CSTATE_HALTED
#define E1701_CSTATE_WRITING_DATA         E170X_CSTATE_WRITING_DATA
#define E1701_CSTATE_WRITING_DATA_ERROR   E170X_CSTATE_WRITING_DATA_ERROR
#define E1701_CSTATE_WAS_MOTION_STARTED   E170X_CSTATE_WAS_MOTION_STARTED
#define E1701_CSTATE_WAS_MOTION_STOPPED   E170X_CSTATE_WAS_MOTION_STOPPED
#define E1701_CSTATE_IS_REFERENCING       E170X_CSTATE_IS_REFERENCING // not used by E1701A/D but by E1701C, defined here due to common firmware use
#define E1701_CSTATE_WAIT_INPUT           E170X_CSTATE_WAIT_INPUT
#define E1701_CSTATE_SAC_READY            E170X_CSTATE_SAC_READY // in stand alone-mode only: similar to DOut0, signals "ready for marking"
#define E1701_CSTATE_SAC_MARKING          E170X_CSTATE_SAC_MARKING // in stand alone-mode only: similar to DOut1, signals "marking active"
#define E1701_CSTATE_SAC_CTLXY            E170X_CSTATE_SAC_CTLXY // in stand alone-mode only: command ctlxy was sent and laser is on until ExtStart is released

#define E1701_MAX_CORRECTION_TABLES E170X_MAX_CORRECTION_TABLES

#define E1701_FILEMODE_OFF   E170X_FILEMODE_OFF
#define E1701_FILEMODE_LOCAL E170X_FILEMODE_LOCAL
#define E1701_FILEMODE_SEND  E170X_FILEMODE_SEND

#define E1701_BSTATE_XY2_100_BB        E170X_BSTATE_XY2_100_BB
#define E1701_BSTATE_ILDA_BB           E170X_BSTATE_ILDA_BB
//#define E1701_BSTATE_CNC_BB            E170X_BSTATE_CNC_BB
#define E1701_BSTATE_LP8_EB            E170X_BSTATE_LP8_EB
#ifndef E1701_BSTATE_DIGI_EB
 #define E1701_BSTATE_DIGI_EB          E170X_BSTATE_DIGI_EB
#endif
#define E1701_BSTATE_LY001_BB          E170X_BSTATE_LY001_BB

#define E1701_BSTATE_BB_MASK           E170X_BSTATE_BB_MASK

#define E1701_TUNE_EXTTRIG_DIGIIN7     E170X_TUNE_EXTTRIG_DIGIIN7
#define E1701_TUNE_2D_MOTF             E170X_TUNE_2D_MOTF
#define E1701_TUNE_SAVE_SERIAL_STATES  E170X_TUNE_SAVE_SERIAL_STATES
#define E1701_TUNE_INVERT_LASERGATE    E170X_TUNE_INVERT_LASERGATE
#define E1701_TUNE_INVERT_LASERA       E170X_TUNE_INVERT_LASERA
#define E1701_TUNE_INVERT_LASERB       E170X_TUNE_INVERT_LASERB
#define E1701_TUNE_LASERA_GPO          E170X_TUNE_LASERA_GPO
#define E1701_TUNE_LASERB_GPO          E170X_TUNE_LASERB_GPO
#define E1701_TUNE_10V_ANALOGUE_XYZ    E170X_TUNE_10V_ANALOGUE_XYZ
#define E1701_TUNE_USE_A1_AS_Z         E170X_TUNE_USE_A1_AS_Z
#define E1701_TUNE_STUPI2D_XY2         E170X_TUNE_STUPI2D_XY2
#define E1701_TUNE_FAST_MATRIX         E170X_TUNE_FAST_MATRIX
#define E1701_TUNE_XY2_18BIT           E170X_TUNE_XY2_18BIT
#define E1701_TUNE_XY3_20BIT           E170X_TUNE_XY3_20BIT
#define E1701_TUNE_DISABLE_TEST        E170X_TUNE_DISABLE_TEST
#define E1701_TUNE_INVERT_MIP          E170X_TUNE_INVERT_MIP
#define E1701_TUNE_INVERT_WET          E170X_TUNE_INVERT_WET
#define E1701_TUNE_EXTTRIG_DIGIIN6     E170X_TUNE_EXTTRIG_DIGIIN6
#define E1701_TUNE_INVERT_EXTSTOP      E170X_TUNE_INVERT_EXTSTOP
#define E1701_TUNE_HALT_WITH_EXTSTART  E170X_TUNE_HALT_WITH_EXTSTART
#define E1701_TUNE_INVERT_LP8          E170X_TUNE_INVERT_LP8
#define E1701_TUNE_INVERT_MO           E170X_TUNE_INVERT_MO
#define E1701_TUNE_INVERT_EXTSTART     E170X_TUNE_INVERT_EXTSTART
#define E1701_TUNE_FORCE_TO_ZERO       E170X_TUNE_FORCE_TO_ZERO
#define E1701_TUNE_SCANNERMODE_MASK    E170X_TUNE_SCANNERMODE_MASK

#define E1701_COMMAND_FLAG_STREAM      E170X_COMMAND_FLAG_STREAM
#define E1701_COMMAND_FLAG_DIRECT      E170X_COMMAND_FLAG_DIRECT
#define E1701_COMMAND_FLAG_PASSIVE     E170X_COMMAND_FLAG_PASSIVE

#define E1701_COMMAND_FLAG_WRITE_MASK             E170X_COMMAND_FLAG_WRITE_MASK
#define E1701_COMMAND_FLAG_WRITE_LP8MO            E170X_COMMAND_FLAG_WRITE_LP8MO
#define E1701_COMMAND_FLAG_WRITE_LP8LATCH         E170X_COMMAND_FLAG_WRITE_LP8LATCH
#define E1701_COMMAND_FLAG_WRITE_LASERA_GPO       E170X_COMMAND_FLAG_WRITE_LASERA_GPO
#define E1701_COMMAND_FLAG_WRITE_LASERB_GPO       E170X_COMMAND_FLAG_WRITE_LASERB_GPO
#define E1701_COMMAND_FLAG_WRITE_LASERGATE        E170X_COMMAND_FLAG_WRITE_LASERGATE
#define E1701_COMMAND_FLAG_WRITE_SPOTSIZE         E170X_COMMAND_FLAG_WRITE_SPOTSIZE

#define E1701_COMMAND_FLAG_DYNDATA_MARK_FONTENTRY E170X_COMMAND_FLAG_DYNDATA_MARK_FONTENTRY

#define E1701_COMMAND_FLAG_MOTF_WAIT_INCS         E170X_COMMAND_FLAG_MOTF_WAIT_INCS
#define E1701_COMMAND_FLAG_MOTF_WAIT_BITS         E170X_COMMAND_FLAG_MOTF_WAIT_BITS

#define E1701_COMMAND_FLAG_XYCORR_FLIPXY          E170X_COMMAND_FLAG_XYCORR_FLIPXY
#define E1701_COMMAND_FLAG_XYCORR_MIRRORX         E170X_COMMAND_FLAG_XYCORR_MIRRORX
#define E1701_COMMAND_FLAG_XYCORR_MIRRORY         E170X_COMMAND_FLAG_XYCORR_MIRRORY

#define E1701_COMMAND_FLAG_SCANNER_VAR_POLYDELAY  E170X_COMMAND_FLAG_SCANNER_VAR_POLYDELAY

#define E1701_PIXELMODE_NO_JUMPS           E170X_PIXELMODE_NO_JUMPS
#define E1701_PIXELMODE_JUMP_N_SHOOT       E170X_PIXELMODE_JUMP_N_SHOOT
#define E1701_PIXELMODE_HW_POWER_CONTROL   E170X_PIXELMODE_HW_POWER_CONTROL
#define E1701_PIXELMODE_GATE_POWER_CONTROL E170X_PIXELMODE_GATE_POWER_CONTROL

#define E1701_FREE_SPACE_PRIMARY   E170X_FREE_SPACE_PRIMARY
#define E1701_FREE_SPACE_SECONDARY E170X_FREE_SPACE_SECONDARY

#ifdef __cplusplus
extern "C"
{
#endif
   typedef int (*E1701_power_callback)(unsigned char n,double power,void *userData); /** type definition for the callback function that has to be provided by client for setting power */
   typedef int (*E1701_progress_callback)(unsigned char n,unsigned int progress,void *userData); /** type definition for the callback function that has to be provided by client for getting progress information */
#ifdef __cplusplus
};
#endif

#ifndef ENV_E1701
 #define E1701_API E170X_API

#ifdef __cplusplus
extern "C"
{
#else
   typedef char bool; 
#endif
   // ***** E1701 easy interface functions *************************************************************************************
   // ***** DEPRECATED, use E170X_...()-functions instead! *********************************************************************
   // base functions
   E1701_API unsigned char E1701_set_connection(const char *address);
   E1701_API unsigned char E1701_set_connection_(const char *address);
   E1701_API void          E1701_set_password(const unsigned char n,const char *ethPwd);
   E1701_API int           E1701_set_filepath(unsigned char n,const char *fname,unsigned int mode);
   E1701_API int           E1701_set_logfile(unsigned char n,const char *path); // DEPRECATED;
   E1701_API int           E1701_set_debug_logfile(const unsigned char n,const char *path,const unsigned char flags); // for logging of local function calls, suitable for debugging of own application
   E1701_API int           E1701_write_debug_logfile(const unsigned char n,const char *format,...); // for writing own debug texts into log
   E1701_API void          E1701_close(unsigned char n);
   E1701_API int           E1701_load_correction(unsigned char n, const char* filename,unsigned char tableNum);
   E1701_API int           E1701_switch_correction(unsigned char n,unsigned char tableNum);
   E1701_API int           E1701_set_xy_correction(unsigned char n, double gainX, double gainY,double rot,int offsetX,int offsetY);
   E1701_API int           E1701_set_xy_correction2(const unsigned char n, const double gainX, const double gainY, const double rot,const int offsetX, const int offsetY,const double slantX, const double slantY); // DEPRECATED, use E1701_set_xy_correction3 instead!
   E1701_API int           E1701_set_xy_correction3(const unsigned char n,const unsigned int flags,const double gainX, const double gainY, const double rot, const int offsetX, const int offsetY, const double slantX, const double slantY);
   E1701_API int           E1701_set_z_correction(const unsigned char n,const unsigned int h,const double xy_to_z_ratio,const int res);
   E1701_API int           E1701_tune(const unsigned char n,const unsigned int tuneFlags);
   E1701_API int           E1701_set_speeds(unsigned char n, double jumpspeed,double markspeed);
   E1701_API int           E1701_set_overspeed(const unsigned char n,const unsigned int flags,const double scannerLag,const double jumpFactor,const double reserved);
   E1701_API int           E1701_set_laser_delays(unsigned char n,double ondelay,double offdelay);
   E1701_API int           E1701_set_laser_mode(unsigned char n, unsigned int mode);
   E1701_API int           E1701_set_laser(const unsigned char n,const unsigned int flags,const char on);
   E1701_API int           E1701_set_wobble(unsigned char n,unsigned int x,unsigned int y,double freq);
   E1701_API int           E1701_set_scanner_delays(unsigned char n,double jumpdelay,double markdelay,double polydelay);
   E1701_API int           E1701_set_scanner_delays2(const unsigned char n,const unsigned int flags,const double jumpdelay,const double markdelay,const double polydelay);
   E1701_API int           E1701_jump_abs(unsigned char n,int x,int y,int z);
   E1701_API int           E1701_mark_abs(unsigned char n,int x,int y,int z);
   E1701_API int           E1701_set_pixelmode(const unsigned char n,const unsigned int mode,const double powerThres,const unsigned int res);
   E1701_API int           E1701_mark_pixelline(const unsigned char n,int x,int y,int z,const int pixWidth,const int pixHeight,const int pixDepth,unsigned int pixNum,const double *pixels,E1701_power_callback power_callback,void *userData);
   E1701_API int           E1701_set_pos(unsigned char n,int x,int y,int z,unsigned char laserOn);
   E1701_API int           E1701_set_matrix(unsigned char n, double m11, double m12, double m21, double m22);
   E1701_API int           E1701_set_matrix2(const unsigned char n,const unsigned int flags,const double m11,const double m12,const double m21,const double m22);
   E1701_API int           E1701_set_trigger_point(unsigned char n);
   E1701_API int           E1701_release_trigger_point(unsigned char n);
   E1701_API int           E1701_set_sync(const unsigned char n,const unsigned int flags,const unsigned int value);
   E1701_API unsigned int  E1701_get_sync(const unsigned char n);
   E1701_API int           E1701_set_extstart(const unsigned char n,const unsigned int flags,const unsigned int receivedCtr,const unsigned int expectedCtr);
   E1701_API int           E1701_get_extstart(const unsigned char n,unsigned int *receivedCtr,unsigned int *expectedCtr);
   E1701_API int           E1701_execute(unsigned char n);
   E1701_API int           E1701_stop_execution(unsigned char n);
   E1701_API int           E1701_halt_execution(unsigned char n,unsigned char halt);
   E1701_API int           E1701_delay(unsigned char n,double delay);
   E1701_API int           E1701_dynamic_data(unsigned char n,struct oapc_bin_struct_dyn_data *dynData);
   E1701_API int           E1701_dynamic_mark(const unsigned char n,const unsigned int flags,const void *value);
   E1701_API int           E1701_loop(const unsigned char n,const unsigned int flags,const unsigned int repeat);
   E1701_API unsigned int  E1701_get_startstop_state(unsigned char n);
   E1701_API unsigned int  E1701_get_card_state(unsigned char n);
   E1701_API int           E1701_get_card_state2(const unsigned char n, unsigned int *state);
   E1701_API unsigned int  E1701_get_card_info(unsigned char n);
   E1701_API int           E1701_set_laser_timing(unsigned char n,double frequency,double pulse);
   E1701_API int           E1701_set_laserb(const unsigned char n,const double frequency,const double pulse);
   E1701_API int           E1701_set_standby(unsigned char n,double frequency,double pulse);
   E1701_API int           E1701_set_standby2(const unsigned char n,const double frequency,const double pulse,const bool force);
   E1701_API int           E1701_set_fpk(unsigned char n,double fpk,double yag3QTime);
   E1701_API int           E1701_set_sky_params(unsigned char n,double angle, unsigned int fadeIn,unsigned int fadeOut);
   E1701_API int           E1701_get_free_space(unsigned char n,int buffer);
   E1701_API void          E1701_get_version(unsigned char n,unsigned short *hwVersion,unsigned short *fwVersion);
   E1701_API int           E1701_get_library_version();
   E1701_API int           E1701_get_serial_number(const unsigned char n,char *serial,const int length);
   E1701_API int           E1701_write(unsigned char n,unsigned int flags,unsigned int value);
   E1701_API int           E1701_repeat(const unsigned char n,const unsigned int repeat);

   // LP8 extension board functions
   E1701_API int           E1701_lp8_write(unsigned char n,unsigned char value);
   E1701_API int           E1701_lp8_write2(unsigned char n,unsigned int flags,unsigned char value);
   E1701_API int           E1701_lp8_write_latch(unsigned char n,unsigned char on,double delay1,unsigned char value,double delay2,double delay3);
   E1701_API int           E1701_lp8_a0(unsigned char n,unsigned char value);
   E1701_API int           E1701_lp8_a0_2(unsigned char n,unsigned int flags,unsigned char value);
   E1701_API int           E1701_lp8_write_mo(unsigned char n,unsigned char on);
   E1701_API int           E1701_lp8_write_mo2(const unsigned char n,const unsigned int flags,const unsigned char on);

   // DIGI I/O extension board functions
   E1701_API int           E1701_digi_write(unsigned char n,unsigned int value);
   E1701_API int           E1701_digi_write2(unsigned char n,unsigned int flags,unsigned int value,unsigned int mask);
   E1701_API int           E1701_digi_pulse(const unsigned char n,const unsigned int flags,const unsigned int in_value,const unsigned int mask,const unsigned int pulses,const double delayOn,const double delayOff);
   E1701_API unsigned int  E1701_digi_read(unsigned char n);
   E1701_API int           E1701_digi_read2(const unsigned char n,unsigned int *value);
   E1701_API int           E1701_digi_read3(const unsigned char n,const unsigned int flags,unsigned int *value);
   E1701_API int           E1701_digi_wait(unsigned char n,unsigned long value,unsigned long mask);
   E1701_API int           E1701_digi_set_motf(unsigned char n,double motfX,double motfY);
   E1701_API int           E1701_digi_set_motf_sim(unsigned char n,double motfX,double motfY);
   E1701_API int           E1701_digi_wait_motf(const unsigned char n,const unsigned int flags,const double dist);
   E1701_API int           E1701_digi_set_mip_output(unsigned char n,unsigned int value,unsigned int flags);
   E1701_API int           E1701_digi_set_wet_output(const unsigned char n,const unsigned int value,const unsigned int flags);

   // Analogue baseboard specific functions
   E1701_API int           E1701_ana_a123(const unsigned char n,const unsigned short r,const unsigned short g,const unsigned short b);

   // Miscellaneous internal-only functions
   E1701_API unsigned int  E1701_send_data(unsigned char n,unsigned int flags,const char *sendData,unsigned int length,unsigned int *sentLength); // DEPRECATED!
   E1701_API unsigned int  E1701_send_data2(const unsigned char n,const unsigned int flags,const char *sendData,unsigned int length,unsigned int *sentLength,E1701_progress_callback progress_callback, void *userData,int *error);
   E1701_API unsigned int  E1701_recv_data(unsigned char n,unsigned int flags,char *recvData,unsigned int maxLength);

   // ***** end of deprecated E1701 easy interface functions *******************************************************************
#ifdef __cplusplus
};
#endif // __cplusplus
#endif // ENV_E1701

#endif //E1701_H
