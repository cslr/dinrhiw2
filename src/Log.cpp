/*
 * Log.cpp
 *
 *  Created on: 14.6.2015
 *      Author: Tomas Ukkonen
 */

#include "Log.h"
#include <stdexcept>
#include <exception>
#include <chrono>
#include <mutex>

#include <stdio.h>
#include <assert.h>


namespace whiteice {

  // program using logging must set output file using setOutputFile()
  Log logging; 

  
  Log::Log()
  {
    std::lock_guard<std::mutex> lock(file_lock);
    t0 = clock::now();
    
    handle = nullptr;
  }
  
  Log::Log(const std::string logFilename)
  {
    {
      std::lock_guard<std::mutex> lock(file_lock);
      
      t0 = clock::now();
      
      handle = nullptr;

      if(logFilename.compare("<stdout>") == 0){
	handle = stdout;
      }
      else{
	handle = fopen(logFilename.c_str(), "wt");
	
	if(handle == 0)
	  fprintf(stderr,
		  "F: Starting logging mechanism failed: %s\n",
		  logFilename.c_str());
      }
    }
    
    info("Logging system started..");
  }
  
  Log::~Log() {
    info("Shutdown? Logging class destroyed.");

    std::lock_guard<std::mutex> lock(file_lock);

    if(handle && handle != stdout)
      fclose(handle);
    
    handle = nullptr;
  }
  
  
  bool Log::setOutputFile(const std::string logFilename)
  {
    if(logFilename.compare("<stdout>") == 0){
      {
	std::lock_guard<std::mutex> lock(file_lock);
	
	if(handle != 0 && handle != stdout) fclose(handle);
	handle = stdout;
      }
    }
    else{
      auto new_handle = fopen(logFilename.c_str(), "wt");
      if(new_handle == nullptr) return false;

      {
	std::lock_guard<std::mutex> lock(file_lock);
	
	if(handle != 0 && handle != stdout) fclose(handle);
	handle = new_handle;
      }
    }
      
    info("Logging system (re)started..");

    return true;
  }


  void Log::setPrintOutput(bool print_stdout)
  {
	  printStdoutToo = print_stdout;
  }


  // logging
  void Log::info(const std::string msg){
    std::lock_guard<std::mutex> lock(file_lock);
    double ms = std::chrono::duration_cast<milliseconds>(clock::now() - t0).count()/1000.0;
    snprintf(buffer, BUFLEN, "INFO %5.5f %s\n", ms, msg.c_str());
    if(handle){
      fputs(buffer, handle);
      fflush(handle);
    }

    if(printStdoutToo){
    	printf(buffer);
    	fflush(stdout);
    }
  }
  
  
  void Log::warn(const std::string msg){
    std::lock_guard<std::mutex> lock(file_lock);
    double ms = std::chrono::duration_cast<milliseconds>(clock::now() - t0).count()/1000.0;
    snprintf(buffer, BUFLEN, "WARN %5.5f %s\n", ms, msg.c_str());
    if(handle){
      fputs(buffer, handle);
      fflush(handle);
    }

    if(printStdoutToo){
        printf(buffer);
        fflush(stdout);
    }
  }
  
  void Log::error(const std::string msg){
    std::lock_guard<std::mutex> lock(file_lock);
    double ms = std::chrono::duration_cast<milliseconds>(clock::now() - t0).count()/1000.0;
    snprintf(buffer, BUFLEN, "ERRO %5.5f %s\n", ms, msg.c_str());
    if(handle){
      fputs(buffer, handle);
      fflush(handle);
    }

    if(printStdoutToo){
        printf(buffer);
        fflush(stdout);
    }
  }
  
  void Log::fatal(const std::string msg){
    std::lock_guard<std::mutex> lock(file_lock);
    double ms = std::chrono::duration_cast<milliseconds>(clock::now() - t0).count()/1000.0;
    snprintf(buffer, BUFLEN, "FATA %5.5f %s\n", ms, msg.c_str());
    if(handle){
      fputs(buffer, handle);
      fflush(handle);
    }

    if(printStdoutToo){
        printf(buffer);
        fflush(stdout);
    }
  }
  
  
} /* namespace whiteice */
