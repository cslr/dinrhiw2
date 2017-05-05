/*
 * Log.h
 *
 *  Created on: 14.6.2015
 *      Author: Tomas Ukkonen
 */

#ifndef LOG_H_
#define LOG_H_

#include <iostream>
#include <string>
#include <stdio.h>

#include <chrono>
#include <mutex>

namespace whiteice {

  class Log {
  public:
    Log();
    Log(std::string logFilename);
    virtual ~Log();
    
    bool setOutputFile(std::string logFilename);
    
    // logging
    void info(std::string msg);
    void warn(std::string msg);
    void error(std::string msg);
    void fatal(std::string msg);
    
    
  protected:
    typedef std::chrono::high_resolution_clock clock;
    typedef std::chrono::milliseconds milliseconds;
    
    FILE* handle = nullptr; // initially prints to stdout until initialized by ctor
    static const int BUFLEN = 65536; // 64 KB
    char buffer[BUFLEN];
    
    clock::time_point t0;
    std::mutex file_lock;
  };

  // program using logging must set output file using setOutputFile()
  extern Log logging;
  
} /* namespace whiteice */

#endif /* LOG_H_ */
