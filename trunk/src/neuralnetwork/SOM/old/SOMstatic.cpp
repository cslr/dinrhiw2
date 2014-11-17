/*
 * static part of implementation of SOM =
 *  static variables of SOM
 */

#include "SOMstatic.h"
#include <string>

namespace whiteice
{  
  // constant field names in SOM configuration files
  const std::string SOM_VERSION_CFGSTR  = "SOM_CONFIG_VERSION";
  const std::string SOM_SIZES_CFGSTR    = "SOM_SIZES";
  const std::string SOM_PARAMS_CFGSTR   = "SOM_PARAMS";
  const std::string SOM_ETA_CFGSTR      = "SOM_USE_ETA";
  const std::string SOM_ROWPROTO_CFGSTR = "SOM_ROW%d";
  
};
