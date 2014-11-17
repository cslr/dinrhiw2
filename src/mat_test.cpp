/*
 * memory access timing border find test
 * test how well border/bad finding works. and check for bugs
 */

#include <iostream>
#include <vector>
#include "memory_access_timing.h"

using namespace std;
 
int main()
{
		//double a = 1;
		//int v = 100;
		//
		//cout << " " << pdf_t(-1,v) << " " << pdf_t(0, v) << " " << pdf_t(1,v) << endl;
		//
		//for(a=0.5;a<3;a+=0.1)
		//		cout << "F(T > " << a << "," << v << ") = " << 1 - pmf_t(a,v) << endl;
		
		
		std::vector<int> borders;
		std::vector<double> access_times;
		
		if(!memory_access_timing(borders, access_times)) return -1;
				
		cout << "MEMORY ACCESS TIME CHANGES: " << endl;
		
		for(unsigned int i=0;i<borders.size();i++)
		{
				cout << "BORDER " << i << "  ";
				cout << borders[i]*4 << " BYTES" << "\t";
				
				if(i != 0){
						if(access_times[i-1] != 0){
								cout << (int)(access_times[i] - access_times[i-1])/access_times[i-1] << "%";
						}
				}
				
				cout << endl;
		}
		
		return 0;
}

