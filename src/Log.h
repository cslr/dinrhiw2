/*
 * Log.h
 *
 *  Created on: 14.6.2015
 *      Author: Tomas Ukkonen
 */

#ifndef LOG_H_
#define LOG_H_

#include <mutex>

#include <iostream>
#include <string>
#include <stdio.h>

#include <chrono>



namespace whiteice {

  class Log {
  public:
    Log();
    Log(const std::string logFilename);
    virtual ~Log();
    
    bool setOutputFile(const std::string logFilename);
    
    void setPrintOutput(bool print_stdout);
    bool getPrintOutput(){ return printStdoutToo; }

    // logging
    void info(const std::string msg);
    void warn(const std::string msg);
    void error(const std::string msg);
    void fatal(const std::string msg);
    
    
  protected:
    typedef std::chrono::high_resolution_clock clock;
    typedef std::chrono::milliseconds milliseconds;
    
    FILE* handle = nullptr; // initially prints to stdout until initialized by ctor
    static const int BUFLEN = 65536; // 64 KB
    char buffer[BUFLEN];
    
    bool printStdoutToo = false;

    clock::time_point t0;
    std::mutex file_lock;
  };

  // program using logging must set output file using setOutputFile()
  extern Log logging;
  
} /* namespace whiteice */

#endif /* LOG_H_ */
