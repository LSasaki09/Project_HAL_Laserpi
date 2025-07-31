#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "libe1701/libe1701.h"

namespace py = pybind11;


void define_constants(py::module_ &m) {
    // Error codes
    m.attr("E170X_OK") = py::int_(E170X_OK);
    m.attr("E170X_ERROR_INVALID_CARD") = py::int_(E170X_ERROR_INVALID_CARD);
    m.attr("E170X_ERROR_NO_CONNECTION") = py::int_(E170X_ERROR_NO_CONNECTION);
    m.attr("E170X_ERROR_NO_MEMORY") = py::int_(E170X_ERROR_NO_MEMORY);
    m.attr("E170X_ERROR_UNKNOWN_FW") = py::int_(E170X_ERROR_UNKNOWN_FW);
    m.attr("E170X_ERROR") = py::int_(E170X_ERROR);

     m.attr("E170X_ERROR_TRANSMISSION") = py::int_(E170X_ERROR_TRANSMISSION);
    m.attr("E170X_ERROR_FILEOPEN") = py::int_(E170X_ERROR_FILEOPEN);
    m.attr("E170X_ERROR_FILEWRITE") = py::int_(E170X_ERROR_FILEWRITE);
    m.attr("E170X_ERROR_BORD_NA") = py::int_(E170X_ERROR_BORD_NA);
    m.attr("E170X_ERROR_INVALID_DATA") = py::int_(E170X_ERROR_INVALID_DATA);
    m.attr("E170X_ERROR_UNKNOWN_BOARD") = py::int_(E170X_ERROR_UNKNOWN_BOARD);
    m.attr("E170X_ERROR_FILENAME") = py::int_(E170X_ERROR_FILENAME);
    m.attr("E170X_ERROR_NOT_SUPPORTED") = py::int_(E170X_ERROR_NOT_SUPPORTED);
    m.attr("E170X_ERROR_NO_DATA_AVAILABLE") = py::int_(E170X_ERROR_NO_DATA_AVAILABLE);
    m.attr("E170X_ERROR_FILEREAD") = py::int_(E170X_ERROR_FILEREAD);
    m.attr("E170X_ERROR_STILL_IN_PROGRESS") = py::int_(E170X_ERROR_STILL_IN_PROGRESS);

    // Configuration constants
    m.attr("E170X_MAX_HEAD_NUM") = py::int_(E170X_MAX_HEAD_NUM);

    // Laser constants
    m.attr("E170X_LASER_FPK") = py::int_(E170X_LASER_FPK);
    m.attr("E170X_LASER_FREQ_ON1") = py::int_(E170X_LASER_FREQ_ON1);
    m.attr("E170X_LASER_FREQ_ON2") = py::int_(E170X_LASER_FREQ_ON2);
    m.attr("E170X_LASER_FREQ_OFF") = py::int_(E170X_LASER_FREQ_OFF);
    m.attr("E170X_LASER_FREQ_DUAL") = py::int_(E170X_LASER_FREQ_DUAL);

    // Laser modes
    m.attr("E170X_LASERMODE_CO2") = py::int_(E170X_LASERMODE_CO2);
    m.attr("E170X_LASERMODE_YAG1") = py::int_(E170X_LASERMODE_YAG1);
    m.attr("E170X_LASERMODE_YAG2") = py::int_(E170X_LASERMODE_YAG2);
    m.attr("E170X_LASERMODE_YAG3") = py::int_(E170X_LASERMODE_YAG3);
    m.attr("E170X_LASERMODE_CRF") = py::int_(E170X_LASERMODE_CRF);
    m.attr("E170X_LASERMODE_DFREQ") = py::int_(E170X_LASERMODE_DFREQ);
    m.attr("E170X_LASERMODE_unused") = py::int_(E170X_LASERMODE_unused);
    m.attr("E170X_LASERMODE_MOPA") = py::int_(E170X_LASERMODE_MOPA);

    // Controller states
    m.attr("E170X_CSTATE_MARKING") = py::int_(E170X_CSTATE_MARKING);
    m.attr("E170X_CSTATE_START_PRESSED") = py::int_(E170X_CSTATE_START_PRESSED);
    m.attr("E170X_CSTATE_STOP_PRESSED") = py::int_(E170X_CSTATE_STOP_PRESSED);
    m.attr("E170X_CSTATE_WAS_START_PRESSED") = py::int_(E170X_CSTATE_WAS_START_PRESSED);
    m.attr("E170X_CSTATE_WAS_STOP_PRESSED") = py::int_(E170X_CSTATE_WAS_STOP_PRESSED);
    m.attr("E170X_CSTATE_ERROR") = py::int_(E170X_CSTATE_ERROR);
    m.attr("E170X_CSTATE_WAS_EXTTRIGGER") = py::int_(E170X_CSTATE_WAS_EXTTRIGGER);
    m.attr("E170X_CSTATE_PROCESSING") = py::int_(E170X_CSTATE_PROCESSING);
    m.attr("E170X_CSTATE_EMITTING") = py::int_(E170X_CSTATE_EMITTING);
    m.attr("E170X_CSTATE_FILE_WRITE_ERROR") = py::int_(E170X_CSTATE_FILE_WRITE_ERROR);
    m.attr("E170X_CSTATE_WAIT_EXTTRIGGER") = py::int_(E170X_CSTATE_WAIT_EXTTRIGGER);
    m.attr("E170X_CSTATE_WAS_SILENTTRIGGER") = py::int_(E170X_CSTATE_WAS_SILENTTRIGGER);
    m.attr("E170X_CSTATE_FILEMODE_ACTIVE") = py::int_(E170X_CSTATE_FILEMODE_ACTIVE);
    m.attr("E170X_CSTATE_HALTED") = py::int_(E170X_CSTATE_HALTED);
    m.attr("E170X_CSTATE_WRITING_DATA") = py::int_(E170X_CSTATE_WRITING_DATA);
    m.attr("E170X_CSTATE_WRITING_DATA_ERROR") = py::int_(E170X_CSTATE_WRITING_DATA_ERROR);
    m.attr("E170X_CSTATE_WAS_MOTION_STARTED") = py::int_(E170X_CSTATE_WAS_MOTION_STARTED);
    m.attr("E170X_CSTATE_WAS_MOTION_STOPPED") = py::int_(E170X_CSTATE_WAS_MOTION_STOPPED);
    m.attr("E170X_CSTATE_IS_REFERENCING") = py::int_(E170X_CSTATE_IS_REFERENCING);
    m.attr("E170X_CSTATE_unused2") = py::int_(E170X_CSTATE_unused2);
    m.attr("E170X_CSTATE_WAIT_INPUT") = py::int_(E170X_CSTATE_WAIT_INPUT);
    m.attr("E170X_CSTATE_SAC_READY") = py::int_(E170X_CSTATE_SAC_READY);
    m.attr("E170X_CSTATE_SAC_MARKING") = py::int_(E170X_CSTATE_SAC_MARKING);
    m.attr("E170X_CSTATE_SAC_CTLXY") = py::int_(E170X_CSTATE_SAC_CTLXY);
    m.attr("E170X_CSTATE_WAIT_EXTSIGNAL") = py::int_(E170X_CSTATE_WAIT_EXTSIGNAL);

    // Correction tables
    m.attr("E170X_MAX_CORRECTION_TABLES") = py::int_(E170X_MAX_CORRECTION_TABLES);

    // File modes
    m.attr("E170X_FILEMODE_OFF") = py::int_(E170X_FILEMODE_OFF);
    m.attr("E170X_FILEMODE_LOCAL") = py::int_(E170X_FILEMODE_LOCAL);
    m.attr("E170X_FILEMODE_SEND") = py::int_(E170X_FILEMODE_SEND);

    // Board states
    m.attr("E170X_BSTATE_XY2_100_BB") = py::int_(E170X_BSTATE_XY2_100_BB);
    m.attr("E170X_BSTATE_ILDA_BB") = py::int_(E170X_BSTATE_ILDA_BB);
    m.attr("E170X_BSTATE_LP8_EB") = py::int_(E170X_BSTATE_LP8_EB);
    m.attr("E170X_BSTATE_DIGI_EB") = py::int_(E170X_BSTATE_DIGI_EB);
    m.attr("E170X_BSTATE_LY001_BB") = py::int_(E170X_BSTATE_LY001_BB);
    m.attr("E170X_BSTATE_BB_MASK") = py::int_(E170X_BSTATE_BB_MASK);

    // Tune flags
    m.attr("E170X_TUNE_EXTTRIG_DIGIIN7") = py::int_(E170X_TUNE_EXTTRIG_DIGIIN7);
    m.attr("E170X_TUNE_2D_MOTF") = py::int_(E170X_TUNE_2D_MOTF);
    m.attr("E170X_TUNE_SAVE_SERIAL_STATES") = py::int_(E170X_TUNE_SAVE_SERIAL_STATES);
    m.attr("E170X_TUNE_INVERT_LASERGATE") = py::int_(E170X_TUNE_INVERT_LASERGATE);
    m.attr("E170X_TUNE_INVERT_LASERA") = py::int_(E170X_TUNE_INVERT_LASERA);
    m.attr("E170X_TUNE_INVERT_LASERB") = py::int_(E170X_TUNE_INVERT_LASERB);
    m.attr("E170X_TUNE_LASERA_GPO") = py::int_(E170X_TUNE_LASERA_GPO);
    m.attr("E170X_TUNE_LASERB_GPO") = py::int_(E170X_TUNE_LASERB_GPO);
    m.attr("E170X_TUNE_USE_A1_AS_Z") = py::int_(E170X_TUNE_USE_A1_AS_Z);
    m.attr("E170X_TUNE_STUPI2D_XY2") = py::int_(E170X_TUNE_STUPI2D_XY2);
    m.attr("E170X_TUNE_XY2_18BIT") = py::int_(E170X_TUNE_XY2_18BIT);
    m.attr("E170X_TUNE_XY3_20BIT") = py::int_(E170X_TUNE_XY3_20BIT);
    m.attr("E170X_TUNE_DISABLE_TEST") = py::int_(E170X_TUNE_DISABLE_TEST);
    m.attr("E170X_TUNE_INVERT_MIP") = py::int_(E170X_TUNE_INVERT_MIP);
    m.attr("E170X_TUNE_INVERT_WET") = py::int_(E170X_TUNE_INVERT_WET);
    m.attr("E170X_TUNE_EXTTRIG_DIGIIN6") = py::int_(E170X_TUNE_EXTTRIG_DIGIIN6);
    m.attr("E170X_TUNE_INVERT_EXTSTOP") = py::int_(E170X_TUNE_INVERT_EXTSTOP);
    m.attr("E170X_TUNE_HALT_WITH_EXTSTART") = py::int_(E170X_TUNE_HALT_WITH_EXTSTART);
    m.attr("E170X_TUNE_INVERT_LP8") = py::int_(E170X_TUNE_INVERT_LP8);
    m.attr("E170X_TUNE_INVERT_MO") = py::int_(E170X_TUNE_INVERT_MO);
    m.attr("E170X_TUNE_INVERT_EXTSTART") = py::int_(E170X_TUNE_INVERT_EXTSTART);
    m.attr("E170X_TUNE_QUICK_STARTUP") = py::int_(E170X_TUNE_QUICK_STARTUP);
    m.attr("E170X_TUNE_FORCE_TO_ZERO") = py::int_(E170X_TUNE_FORCE_TO_ZERO);
    m.attr("E170X_TUNE_DONOTUSE") = py::int_(E170X_TUNE_DONOTUSE);
    m.attr("E170X_TUNE_SCANNERMODE_MASK") = py::int_(E170X_TUNE_SCANNERMODE_MASK);

    // Command flags
    m.attr("E170X_COMMAND_FLAG_STREAM") = py::int_(E170X_COMMAND_FLAG_STREAM);
    m.attr("E170X_COMMAND_FLAG_DIRECT") = py::int_(E170X_COMMAND_FLAG_DIRECT);
    m.attr("E170X_COMMAND_FLAG_PASSIVE") = py::int_(E170X_COMMAND_FLAG_PASSIVE);
    m.attr("E170X_COMMAND_FLAG_SILENT") = py::int_(E170X_COMMAND_FLAG_SILENT);
    m.attr("E170X_COMMAND_FLAG_HIGH_LEVEL") = py::int_(E170X_COMMAND_FLAG_HIGH_LEVEL);
    m.attr("E170X_COMMAND_FLAG_WRITE_MASK") = py::int_(E170X_COMMAND_FLAG_WRITE_MASK);
    m.attr("E170X_COMMAND_FLAG_WRITE_LP8MO") = py::int_(E170X_COMMAND_FLAG_WRITE_LP8MO);
    m.attr("E170X_COMMAND_FLAG_WRITE_LP8LATCH") = py::int_(E170X_COMMAND_FLAG_WRITE_LP8LATCH);
    m.attr("E170X_COMMAND_FLAG_WRITE_LASERA_GPO") = py::int_(E170X_COMMAND_FLAG_WRITE_LASERA_GPO);
    m.attr("E170X_COMMAND_FLAG_WRITE_LASERB_GPO") = py::int_(E170X_COMMAND_FLAG_WRITE_LASERB_GPO);
    m.attr("E170X_COMMAND_FLAG_WRITE_LASERGATE") = py::int_(E170X_COMMAND_FLAG_WRITE_LASERGATE);
    m.attr("E170X_COMMAND_FLAG_WRITE_SPOTSIZE") = py::int_(E170X_COMMAND_FLAG_WRITE_SPOTSIZE);
    m.attr("E170X_COMMAND_FLAG_WRITE_LASERA_GPO_PULSE") = py::int_(E170X_COMMAND_FLAG_WRITE_LASERA_GPO_PULSE);
    m.attr("E170X_COMMAND_FLAG_DYNDATA_MARK_FONTENTRY") = py::int_(E170X_COMMAND_FLAG_DYNDATA_MARK_FONTENTRY);
    m.attr("E170X_COMMAND_FLAG_MOTF_WAIT_INCS") = py::int_(E170X_COMMAND_FLAG_MOTF_WAIT_INCS);
    m.attr("E170X_COMMAND_FLAG_MOTF_WAIT_BITS") = py::int_(E170X_COMMAND_FLAG_MOTF_WAIT_BITS);
    m.attr("E170X_COMMAND_FLAG_XYCORR_FLIPXY") = py::int_(E170X_COMMAND_FLAG_XYCORR_FLIPXY);
    m.attr("E170X_COMMAND_FLAG_XYCORR_MIRRORX") = py::int_(E170X_COMMAND_FLAG_XYCORR_MIRRORX);
    m.attr("E170X_COMMAND_FLAG_XYCORR_MIRRORY") = py::int_(E170X_COMMAND_FLAG_XYCORR_MIRRORY);
    m.attr("E170X_COMMAND_FLAG_ZCORR_MIRRORZ") = py::int_(E170X_COMMAND_FLAG_ZCORR_MIRRORZ);
    m.attr("E170X_COMMAND_FLAG_SCANNER_VAR_POLYDELAY") = py::int_(E170X_COMMAND_FLAG_SCANNER_VAR_POLYDELAY);
    m.attr("E170X_COMMAND_FLAG_ANA_AOUT0") = py::int_(E170X_COMMAND_FLAG_ANA_AOUT0);
    m.attr("E170X_COMMAND_FLAG_ANA_MASK") = py::int_(E170X_COMMAND_FLAG_ANA_MASK);

    // Pixel modes
    m.attr("E170X_PIXELMODE_NO_JUMPS") = py::int_(E170X_PIXELMODE_NO_JUMPS);
    m.attr("E170X_PIXELMODE_JUMP_N_SHOOT") = py::int_(E170X_PIXELMODE_JUMP_N_SHOOT);
    m.attr("E170X_PIXELMODE_HW_POWER_CONTROL") = py::int_(E170X_PIXELMODE_HW_POWER_CONTROL);
    m.attr("E170X_PIXELMODE_GATE_POWER_CONTROL") = py::int_(E170X_PIXELMODE_GATE_POWER_CONTROL);
    m.attr("E170X_PIXELMODE_JUMP_LEAVE_POWER") = py::int_(E170X_PIXELMODE_JUMP_LEAVE_POWER);

    // Free space constants
    m.attr("E170X_FREE_SPACE_PRIMARY") = py::int_(E170X_FREE_SPACE_PRIMARY);
    m.attr("E170X_FREE_SPACE_SECONDARY") = py::int_(E170X_FREE_SPACE_SECONDARY);
    m.attr("E170X_USED_SPACE_QUEUE") = py::int_(E170X_USED_SPACE_QUEUE);
    
}

PYBIND11_MODULE(libe1701py, m) {
    m.doc() = "Bindings Python pour libe1701";

    define_constants(m);

    m.def(
        "set_connection",
        &E1701_set_connection,
        py::arg("address"),
        "Définit la connexion à l'adresse spécifiée (retourne un code de statut unsigned char)"
    );

    m.def(
        "set_connection_",
        &E1701_set_connection_,
        py::arg("address"),
        "Définit une variante de la connexion à l'adresse spécifiée (retourne un code de statut unsigned char)"
    );

    m.def(
        "set_password",
        &E1701_set_password,
        py::arg("n"),
        py::arg("ethPwd"),
        "Définit le mot de passe pour l'appareil"
    );

    m.def(
        "set_filepath",
        &E1701_set_filepath,
        py::arg("n"),
        py::arg("fname"),
        py::arg("mode"),
        "Définit le chemin du fichier pour l'appareil (retourne un code de statut int)"
    );

    m.def(
        "set_logfile",
        &E1701_set_logfile,
        py::arg("n"),
        py::arg("path"),
        "Définit le chemin du fichier de log (DEPRECATED, retourne un code de statut int)"
    );

    m.def(
        "set_debug_logfile",
        &E1701_set_debug_logfile,
        py::arg("n"),
        py::arg("path"),
        py::arg("flags"),
        "Définit le fichier de log de débogage avec des flags (retourne un code de statut int)"
    );

    m.def(
        "write_debug_logfile",
        [](unsigned char n, const char *format) {
            return E1701_write_debug_logfile(n, format);
        },
        py::arg("n"),
        py::arg("format"),
        "Écrit un texte de débogage dans le fichier de log (retourne un code de statut int, arguments variadiques non pris en charge)"
    );

    m.def(
        "close",
        &E1701_close,
        py::arg("n"),
        "Ferme la connexion à l'appareil"
    );

    m.def(
        "load_correction",
        &E1701_load_correction,
        py::arg("n"),
        py::arg("filename"),
        py::arg("tableNum"),
        "Charge un fichier de correction (retourne un code de statut int)"
    );

    m.def(
        "switch_correction",
        &E1701_switch_correction,
        py::arg("n"),
        py::arg("tableNum"),
        "Change la table de correction active (retourne un code de statut int)"
    );

    m.def(
        "set_xy_correction",
        &E1701_set_xy_correction,
        py::arg("n"),
        py::arg("gainX"),
        py::arg("gainY"),
        py::arg("rot"),
        py::arg("offsetX"),
        py::arg("offsetY"),
        "Définit les paramètres de correction XY (retourne un code de statut int)"
    );

    m.def(
        "set_xy_correction2",
        &E1701_set_xy_correction2,
        py::arg("n"),
        py::arg("gainX"),
        py::arg("gainY"),
        py::arg("rot"),
        py::arg("offsetX"),
        py::arg("offsetY"),
        py::arg("slantX"),
        py::arg("slantY"),
        "Définit les paramètres de correction XY avancés (DEPRECATED, retourne un code de statut int)"
    );

    m.def(
        "set_xy_correction3",
        &E1701_set_xy_correction3,
        py::arg("n"),
        py::arg("flags"),
        py::arg("gainX"),
        py::arg("gainY"),
        py::arg("rot"),
        py::arg("offsetX"),
        py::arg("offsetY"),
        py::arg("slantX"),
        py::arg("slantY"),
        "Définit les paramètres de correction XY avancés avec flags (retourne un code de statut int)"
    );

    m.def(
        "set_z_correction",
        &E1701_set_z_correction,
        py::arg("n"),
        py::arg("h"),
        py::arg("xy_to_z_ratio"),
        py::arg("res"),
        "Définit les paramètres de correction Z (retourne un code de statut int)"
    );

    m.def(
        "tune",
        &E1701_tune,
        py::arg("n"),
        py::arg("tuneFlags"),
        "Règle l'appareil avec des flags (retourne un code de statut int)"
    );

    m.def(
        "set_speeds",
        &E1701_set_speeds,
        py::arg("n"),
        py::arg("jumpspeed"),
        py::arg("markspeed"),
        "Définit les vitesses de saut et de marquage (retourne un codeIci, un wrapper lambda est utilisé pour gérer les paramètres de sortie et retourner un tuple avec le code de retour et les valeurs de sortie."
    );

    m.def(
        "get_version",
        [](unsigned char n) {
            unsigned short hwVersion, fwVersion;
            E1701_get_version(n, &hwVersion, &fwVersion);
            return py::make_tuple(hwVersion, fwVersion);
        },
        py::arg("n"),
        "Obtient les versions matérielle et logicielle (retourne un tuple (hwVersion, fwVersion))"
    );

    m.def(
        "get_card_state2",
        [](unsigned char n) {
            unsigned int state;
            int ret = E1701_get_card_state2(n, &state);
            return py::make_tuple(ret, state);
        },
        py::arg("n"),
        "Obtient l'état de la carte (retourne un tuple (code de retour, état))"
    );

    m.def(
        "get_extstart",
        [](unsigned char n) {
            unsigned int receivedCtr, expectedCtr;
            int ret = E1701_get_extstart(n, &receivedCtr, &expectedCtr);
            return py::make_tuple(ret, receivedCtr, expectedCtr);
        },
        py::arg("n"),
        "Obtient les compteurs de démarrage externe (retourne un tuple (code de retour, receivedCtr, expectedCtr))"
    );

    m.def(
        "get_serial_number",
        [](unsigned char n) {
            char serial[256];
            int ret = E1701_get_serial_number(n, serial, 256);
            return ret == 0 ? std::string(serial) : std::string("");
        },
        py::arg("n"),
        "Obtient le numéro de série sous forme de chaîne (retourne une chaîne vide en cas d'échec)"
    );

    m.def(
        "mark_pixelline",
        [](unsigned char n, int x, int y, int z, int pixWidth, int pixHeight, int pixDepth, unsigned int pixNum, py::array_t<double> pixels) {
            py::buffer_info buf = pixels.request();
            if (buf.ndim != 1) throw std::runtime_error("pixels doit être un tableau 1D");
            if (buf.size < pixNum) throw std::runtime_error("Le tableau pixels est trop petit");
            return E1701_mark_pixelline(n, x, y, z, pixWidth, pixHeight, pixDepth, pixNum, static_cast<const double*>(buf.ptr), nullptr, nullptr);
        },
        py::arg("n"),
        py::arg("x"),
        py::arg("y"),
        py::arg("z"),
        py::arg("pixWidth"),
        py::arg("pixHeight"),
        py::arg("pixDepth"),
        py::arg("pixNum"),
        py::arg("pixels"),
        "Marque une ligne de pixels avec un tableau de pixels (callback non pris en charge, retourne un code de statut int)"
    );

    m.def(
        "set_overspeed",
        &E1701_set_overspeed,
        py::arg("n"),
        py::arg("flags"),
        py::arg("scannerLag"),
        py::arg("jumpFactor"),
        py::arg("reserved"),
        "Définit les paramètres de survitesse (retourne un code de statut int)"
    );

    m.def(
        "set_laser_delays",
        &E1701_set_laser_delays,
        py::arg("n"),
        py::arg("protective_field_ondelay"),
        py::arg("offdelay"),
        "Définit les délais du laser (retourne un code de statut int)"
    );

    m.def(
        "set_laser_mode",
        &E1701_set_laser_mode,
        py::arg("n"),
        py::arg("mode"),
        "Définit le mode du laser (retourne un code de statut int)"
    );

    m.def(
        "set_laser",
        &E1701_set_laser,
        py::arg("n"),
        py::arg("flags"),
        py::arg("on"),
        "Active ou désactive le laser avec des flags (retourne un code de statut int)"
    );

    m.def(
        "set_wobble",
        &E1701_set_wobble,
        py::arg("n"),
        py::arg("x"),
        py::arg("y"),
        py::arg("freq"),
        "Définit les paramètres de wobble (retourne un code de statut int)"
    );

    m.def(
        "set_scanner_delays",
        &E1701_set_scanner_delays,
        py::arg("n"),
        py::arg("jumpdelay"),
        py::arg("markdelay"),
        py::arg("polydelay"),
        "Définit les délais du scanner (retourne un code de statut int)"
    );

    m.def(
        "set_scanner_delays2",
        &E1701_set_scanner_delays2,
        py::arg("n"),
        py::arg("flags"),
        py::arg("jumpdelay"),
        py::arg("markdelay"),
        py::arg("polydelay"),
        "Définit les délais du scanner avec flags (retourne un code de statut int)"
    );

    m.def(
        "jump_abs",
        &E1701_jump_abs,
        py::arg("n"),
        py::arg("x"),
        py::arg("y"),
        py::arg("z"),
        "Effectue un saut absolu (retourne un code de statut int)"
    );

    m.def(
        "mark_abs",
        &E1701_mark_abs,
        py::arg("n"),
        py::arg("x"),
        py::arg("y"),
        py::arg("z"),
        "Marque une position absolue (retourne un code de statut int)"
    );

    m.def(
        "set_pixelmode",
        &E1701_set_pixelmode,
        py::arg("n"),
        py::arg("mode"),
        py::arg("powerThres"),
        py::arg("res"),
        "Définit le mode pixel (retourne un code de statut int)"
    );

    m.def(
        "set_pos",
        &E1701_set_pos,
        py::arg("n"),
        py::arg("x"),
        py::arg("y"),
        py::arg("z"),
        py::arg("laserOn"),
        "Définit une position avec état du laser (retourne un code de statut int)"
    );

    m.def(
        "get_pos",
        [](unsigned char n) {
            int x, y, z;
            int ret = E170X_get_pos(n, &x, &y, &z);
            return py::make_tuple(ret, x, y, z);
        },
        py::arg("n"),
        "Get the last position of the scanhead (returns a tuple (status code, x, y, z))"
    );

    m.def(
        "set_matrix",
        &E1701_set_matrix,
        py::arg("n"),
        py::arg("m11"),
        py::arg("m12"),
        py::arg("m21"),
        py::arg("m22"),
        "Définit une matrice de transformation (retourne un code de statut int)"
    );

    m.def(
        "set_matrix2",
        &E1701_set_matrix2,
        py::arg("n"),
        py::arg("flags"),
        py::arg("m11"),
        py::arg("m12"),
        py::arg("m21"),
        py::arg("m22"),
        "Définit une matrice de transformation avec flags (retourne un code de statut int)"
    );

    m.def(
        "set_trigger_point",
        &E1701_set_trigger_point,
        py::arg("n"),
        "Définit un point de déclenchement (retourne un code de statut int)"
    );

    m.def(
        "release_trigger_point",
        &E1701_release_trigger_point,
        py::arg("n"),
        "Libère un point de déclenchement (retourne un code de statut int)"
    );

    m.def(
        "set_sync",
        &E1701_set_sync,
        py::arg("n"),
        py::arg("flags"),
        py::arg("value"),
        "Définit la synchronisation (retourne un code de statut int)"
    );

    m.def(
        "get_sync",
        &E1701_get_sync,
        py::arg("n"),
        "Obtient l'état de synchronisation (retourne une valeur unsigned int)"
    );

    m.def(
        "set_extstart",
        &E1701_set_extstart,
        py::arg("n"),
        py::arg("flags"),
        py::arg("receivedCtr"),
        py::arg("expectedCtr"),
        "Définit le démarrage externe (retourne un code de statut int)"
    );

    m.def(
        "execute",
        &E1701_execute,
        py::arg("n"),
        "Exécute une commande (retourne un code de statut int)"
    );

    m.def(
        "stop_execution",
        &E1701_stop_execution,
        py::arg("n"),
        "Arrête l'exécution (retourne un code de statut int)"
    );

    m.def(
        "halt_execution",
        &E1701_halt_execution,
        py::arg("n"),
        py::arg("halt"),
        "Met en pause l'exécution (retourne un code de statut int)"
    );

    m.def(
        "delay",
        &E1701_delay,
        py::arg("n"),
        py::arg("delay"),
        "Ajoute un délai (retourne un code de statut int)"
    );

    
    m.def(
        "dynamic_mark",
        [](unsigned char n, unsigned int flags, const void *value) {
            return E1701_dynamic_mark(n, flags, value);
        },
        py::arg("n"),
        py::arg("flags"),
        py::arg("value"),
        "Marque dynamiquement (retourne un code de statut int, value doit être géré séparément)"
    );

    m.def(
        "loop",
        &E1701_loop,
        py::arg("n"),
        py::arg("flags"),
        py::arg("repeat"),
        "Exécute une boucle (retourne un code de statut int)"
    );

    m.def(
        "get_startstop_state",
        &E1701_get_startstop_state,
        py::arg("n"),
        "Obtient l'état de démarrage/arrêt (retourne une valeur unsigned int)"
    );

    m.def(
        "get_card_state",
        &E1701_get_card_state,
        py::arg("n"),
        "Obtient l'état de la carte (retourne une valeur unsigned int)"
    );

    m.def(
        "get_card_info",
        &E1701_get_card_info,
        py::arg("n"),
        "Obtient les informations de la carte (retourne une valeur unsigned int)"
    );

    m.def(
        "set_laser_timing",
        &E1701_set_laser_timing,
        py::arg("n"),
        py::arg("frequency"),
        py::arg("pulse"),
        "Définit le timing du laser (retourne un code de statut int)"
    );

    m.def(
        "set_laserb",
        &E1701_set_laserb,
        py::arg("n"),
        py::arg("frequency"),
        py::arg("pulse"),
        "Définit le timing du laser B (retourne un code de statut int)"
    );

    m.def(
        "set_standby",
        &E1701_set_standby,
        py::arg("n"),
        py::arg("frequency"),
        py::arg("pulse"),
        "Définit le mode veille (retourne un code de statut int)"
    );

    m.def(
        "set_standby2",
        &E1701_set_standby2,
        py::arg("n"),
        py::arg("frequency"),
        py::arg("pulse"),
        py::arg("force"),
        "Définit le mode veille avec forçage (retourne un code de statut int)"
    );

    m.def(
        "set_fpk",
        &E1701_set_fpk,
        py::arg("n"),
        py::arg("fpk"),
        py::arg("yag3QTime"),
        "Définit les paramètres FPK (retourne un code de statut int)"
    );

    m.def(
        "set_sky_params",
        &E1701_set_sky_params,
        py::arg("n"),
        py::arg("angle"),
        py::arg("fadeIn"),
        py::arg("fadeOut"),
        "Définit les paramètres SKY (retourne un code de statut int)"
    );

    m.def(
        "get_free_space",
        &E1701_get_free_space,
        py::arg("n"),
        py::arg("buffer"),
        "Obtient l'espace libre dans le buffer (retourne un int)"
    );

    m.def(
        "get_library_version",
        &E1701_get_library_version,
        "Obtient la version de la bibliothèque (retourne un int)"
    );

    m.def(
        "write",
        &E1701_write,
        py::arg("n"),
        py::arg("flags"),
        py::arg("value"),
        "Écrit une valeur avec des flags (retourne un code de statut int)"
    );

    m.def(
        "repeat",
        &E1701_repeat,
        py::arg("n"),
        py::arg("repeat"),
        "Répète une opération (retourne un code de statut int)"
    );

     // Fonctions pour LP8 Extension Board
    m.def(
        "lp8_write",
        &E1701_lp8_write,
        py::arg("n"), py::arg("value"),
        "Écrit une valeur sur le port LP8 (retourne un code de statut int)"
    );

    m.def(
        "lp8_write2",
        &E1701_lp8_write2,
        py::arg("n"), py::arg("flags"), py::arg("value"),
        "Écrit une valeur sur le port LP8 avec des flags (retourne un code de statut int)"
    );

    m.def(
        "lp8_write_latch",
        &E1701_lp8_write_latch,
        py::arg("n"), py::arg("on"), py::arg("delay1"), py::arg("value"), py::arg("delay2"), py::arg("delay3"),
        "Écrit une valeur sur le latch LP8 avec des délais (retourne un code de statut int)"
    );

    m.def(
        "lp8_a0",
        &E1701_lp8_a0,
        py::arg("n"), py::arg("value"),
        "Écrit une valeur sur le canal A0 de LP8 (retourne un code de statut int)"
    );

    m.def(
        "lp8_a0_2",
        &E1701_lp8_a0_2,
        py::arg("n"), py::arg("flags"), py::arg("value"),
        "Écrit une valeur sur le canal A0 de LP8 avec des flags (retourne un code de statut int)"
    );

    m.def(
        "lp8_write_mo",
        &E1701_lp8_write_mo,
        py::arg("n"), py::arg("on"),
        "Contrôle le signal MO (Main Oscillator) sur LP8 (retourne un code de statut int)"
    );

    m.def(
        "lp8_write_mo2",
        &E1701_lp8_write_mo2,
        py::arg("n"), py::arg("flags"), py::arg("on"),
        "Contrôle le signal MO sur LP8 avec des flags (retourne un code de statut int)"
    );

    // Fonctions pour DIGI I/O Extension Board
    m.def(
        "digi_write",
        &E1701_digi_write,
        py::arg("n"), py::arg("value"),
        "Écrit une valeur sur les sorties numériques (retourne un code de statut int)"
    );

    m.def(
        "digi_write2",
        &E1701_digi_write2,
        py::arg("n"), py::arg("flags"), py::arg("value"), py::arg("mask"),
        "Écrit une valeur sur les sorties numériques avec un masque et des flags (retourne un code de statut int)"
    );

    m.def(
        "digi_pulse",
        &E1701_digi_pulse,
        py::arg("n"), py::arg("flags"), py::arg("in_value"), py::arg("mask"), py::arg("pulses"), py::arg("delayOn"), py::arg("delayOff"),
        "Génère des impulsions sur les sorties numériques (retourne un code de statut int)"
    );

    m.def(
        "digi_read",
        &E1701_digi_read,
        py::arg("n"),
        "Lit l'état des entrées numériques (retourne la valeur lue unsigned int)"
    );

    m.def(
        "digi_read2",
        [](unsigned char n) {
            unsigned int value;
            int ret = E1701_digi_read2(n, &value);
            return py::make_tuple(ret, value);
        },
        py::arg("n"),
        "Lit l'état des entrées numériques (retourne un tuple (code de retour int, valeur unsigned int))"
    );

    m.def(
        "digi_read3",
        [](unsigned char n, unsigned int flags) {
            unsigned int value;
            int ret = E1701_digi_read3(n, flags, &value);
            return py::make_tuple(ret, value);
        },
        py::arg("n"), py::arg("flags"),
        "Lit l'état des entrées numériques avec des flags (retourne un tuple (code de retour int, valeur unsigned int))"
    );

    m.def(
        "digi_wait",
        &E1701_digi_wait,
        py::arg("n"), py::arg("value"), py::arg("mask"),
        "Attend que les entrées numériques correspondent à une valeur avec un masque (retourne un code de statut int)"
    );

    m.def(
        "digi_set_motf",
        &E1701_digi_set_motf,
        py::arg("n"), py::arg("motfX"), py::arg("motfY"),
        "Définit les paramètres MOTF (Marking On The Fly) pour X et Y (retourne un code de statut int)"
    );

    m.def(
        "digi_set_motf_sim",
        &E1701_digi_set_motf_sim,
        py::arg("n"), py::arg("motfX"), py::arg("motfY"),
        "Simule les paramètres MOTF pour X et Y (retourne un code de statut int)"
    );

    m.def(
        "digi_wait_motf",
        &E1701_digi_wait_motf,
        py::arg("n"), py::arg("flags"), py::arg("dist"),
        "Attend une distance MOTF spécifique (retourne un code de statut int)"
    );

    m.def(
        "digi_set_mip_output",
        &E1701_digi_set_mip_output,
        py::arg("n"), py::arg("value"), py::arg("flags"),
        "Définit la sortie MIP (Mark In Progress) (retourne un code de statut int)"
    );

    m.def(
        "digi_set_wet_output",
        &E1701_digi_set_wet_output,
        py::arg("n"), py::arg("value"), py::arg("flags"),
        "Définit la sortie WET (Wait for External Trigger) (retourne un code de statut int)"
    );

    // Fonctions pour Analogue Baseboard
    m.def(
        "ana_a123",
        &E1701_ana_a123,
        py::arg("n"), py::arg("r"), py::arg("g"), py::arg("b"),
        "Définit les valeurs analogiques pour les canaux RGB (retourne un code de statut int)"
    );

    // Fonctions diverses (Miscellaneous)

    m.def(
        "send_data2",
        [](unsigned char n, unsigned int flags, const std::string& sendData) {
            unsigned int sentLength;
            int error;
            unsigned int ret = E1701_send_data2(n, flags, sendData.c_str(), sendData.length(), &sentLength, nullptr, nullptr, &error);
            return py::make_tuple(ret, sentLength, error);
        },
        py::arg("n"), py::arg("flags"), py::arg("sendData"),
        "Envoie des données à l'appareil avec gestion d'erreurs (retourne un tuple (code de retour unsigned int, longueur envoyée unsigned int, erreur int))"
    );

    m.def(
        "recv_data",
        [](unsigned char n, unsigned int flags, unsigned int maxLength) {
            std::vector<char> recvData(maxLength);
            unsigned int ret = E1701_recv_data(n, flags, recvData.data(), maxLength);
            return py::make_tuple(ret, std::string(recvData.data(), ret));
        },
        py::arg("n"), py::arg("flags"), py::arg("maxLength"),
        "Reçoit des données de l'appareil (retourne un tuple (code de retour unsigned int, données reçues string))"
    );

}