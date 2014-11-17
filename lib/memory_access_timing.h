
#ifndef memory_access_timing_h
#define memory_access_timing_h

#include <vector>

/*
 * detects memory hierarchy memory access times statistically
 *
 * saves found memory access time change borders in given vectors
 * returns true if there was no errors (possible zero amount of borders found)
 * returns false in case of failure (vector data unspecified).
 */
bool memory_access_timing(std::vector<int>& memborders,
		std::vector<double>& access_times);


double pmf_t(double t, double v);
double pdf_t(double t, double v);


#endif






