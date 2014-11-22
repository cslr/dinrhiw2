/*
 * simple whiteice::conffile fileformat graph etc.
 * plotting utility
 * 
 * there's some plans to greatly extend this to a real
 * data visualization software.
 * 
 * TODO: 
 *  - add possiblity to ICA set of float/int signals
 *  - visualize text strings by using SOM and 
 *    edit distance between vectors/strings
 */

#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <ctime>

// change to use SDL instead of DLIB

#include <dinrhiw/dinrhiw.h> // core library

#include <string.h>
#include "dataplot.h"

using namespace dlib_global;
using namespace whiteice;

bool init(int argc, char** argv,
	  std::string& filename,
	  std::vector<std::string>& options,
	  std::vector<std::string>& plotsymbols,
	  Font*& font) throw();

void show_usage() throw();
bool has_option(std::vector<std::string>& options, std::string opname) throw();

void show_statistics(const std::vector<int>& ivalues,         Font* font) throw();
void show_statistics(const std::vector<float>& fvalues,       Font* font) throw();
void show_statistics(const std::vector<std::string>& svalues, Font* font) throw();
		     
void show_plot(const std::vector<int>& ivalues,         Font* font) throw();
void show_plot(const std::vector<float>& fvalues,       Font* font) throw();
void show_plot(const std::vector<std::string>& svalues, Font* font) throw();

void show_scatterplot(const std::vector< math::vertex< math::blas_real<float> > >& data, Font* font,
		      const std::vector<std::string>& labels, const std::vector<unsigned int>& labeled);



int main(int argc, char** argv)
{  
  whiteice::conffile datafile;
  Font* font = 0;
  
  std::string filename; // filename for source datafile
  std::vector<std::string> plotsymbols; // symbols to plot
  std::vector<std::string> options;
  
  std::cout << "whiteice data visualization [nop@iki.fi]" << std::endl;
  
  
  if(init(argc, argv, filename, options, plotsymbols , font) == false){
    show_usage();
    return -1;
  }
  
  
  if(datafile.load(filename) == false){
    std::cout << "datafile '" << filename << "' not found." 
	      << std::endl;
    delete font;
    return -1;
  }
  
  
  if(has_option(options, LIST_SYMBOLS_OPTION)){
    std::vector<std::string> symbols;
    if(datafile.get(symbols) == false){
      std::cout << "error: conffile::get() failed." << std::endl;
      delete font;
      return -1;
    }
    
    std::cout << "symbol list: ";
    for(std::vector<std::string>::iterator i=symbols.begin();
	i!=symbols.end();i++)
      std::cout << "'" << *i << "' ";
    
    std::cout << std::endl;
    delete font;
    return 0;
  }
  else if(has_option(options, SCATTERPLOT_OPTION)){
    
    std::vector<std::string> symbols;
    if(datafile.get(symbols) == false){
      std::cout << "error: conffile::get() failed." << std::endl;
      delete font;
      return -1;
    }
    
    std::vector<float> fvalues;
    unsigned int dimension=0;
    
    {
      std::map<unsigned int, unsigned int> dimcount;
      
      // finds most common dimension
      for(unsigned int i=0;i<symbols.size();i++){
	if(datafile.get(symbols[i], fvalues))
	  dimcount[fvalues.size()]++;
	
	fvalues.clear();
      }
      
      
      unsigned int bestcount = 0;
      
      std::map<unsigned int, unsigned int>::iterator i;
      for(i=dimcount.begin();i!=dimcount.end();i++){
	if(i->second > bestcount){
	  dimension = i->first;
	  bestcount = i->second;
	}
      }
      
    }
    
    if(dimension <= 1){
      std::cout << "cannot create scatter plot from 1 dimensional data." << std::endl;
      
      delete font;
      return -1;
    }
    
    
    
    std::vector<math::vertex< math::blas_real<float> > > data;
    std::vector<unsigned int> labeled;
    std::vector<std::string> labels;
    
    // calculates PCA mean removal + whitening    
    {
      // FIXME: doesn't handle outliers correctly
      
      math::vertex< math::blas_real<float> > mean(dimension);
      std::vector<std::string> point_names;
      std::vector<std::string> tmp_labels;
      float varerror;
      
      datafile.get("labeled_points", point_names);
      datafile.get("point_labels", tmp_labels);
      
      if(tmp_labels.size() < point_names.size())
	point_names.resize(tmp_labels.size());
      
      
      for(unsigned int i=0;i<symbols.size();i++){
	math::vertex< math::blas_real<float> > d(dimension);
	
	if(datafile.get(symbols[i], fvalues)){
	  if(fvalues.size() == dimension){
	    for(unsigned int j=0;j<dimension;j++)
	      d[j] = fvalues[j];
	    
	    mean += d;
	    data.push_back(d);
	    
	    for(unsigned int j=0;j<point_names.size();j++){
	      if(symbols[i] == point_names[j]){
		labeled.push_back(data.size()-1);
		labels.push_back(tmp_labels[j]);
		break;
	      }
	    }
	  }
	}
	
      }
      
      mean /= data.size();
      
      // mean removal and correlation matrix calculation
      math::matrix< math::blas_real<float> > R;
      
      for(unsigned int i=0;i<data.size();i++)
	data[i] -= mean;
      
      math::autocorrelation(R, data);
      
      math::matrix< math::blas_real<float> > A(dimension,dimension);

      symmetric_eig(R, A);
      
      // finds the two highest variance dimensions
      
      unsigned int b0, b1;
      float v0, v1;
      
      v0 = R(0,0).value(); b0 = 0;
      v1 = R(1,1).value(); b1 = 1;
      
      if(v0 < v1){
	v0 = v1;
	v1 = R(0,0).value();
	b0 = 1;
	b1 = 0;
      }
      
      for(unsigned int i=2;i<R.xsize();i++){
	if(R(i,i) > v0){
	  v1 = v0;
	  b1 = b0;
	  
	  v0 = R(i,i).value();
	  b1 = i;
	}
	else if(R(i,i) > v1){
	  v1 = R(i,i).value();
	  b1 = i;
	}
      }
      
      // calculates variance that will be lost
      // because of PCA
      
      varerror = 0.0f;
      
      for(unsigned int i=0;i<R.xsize();i++)
	if(i != b0 && i != b1)
	  varerror += R(i,i).value();
      
      // creates decorrelation and dimensionality
      // reduction matrix
      // R = XDX' = E[xx^t] => y = D^-0.5*X'*x, E[yy^t] = I
      
      if(b0 != 0 && b1 != 0){
	for(unsigned int i=0;i<A.xsize();i++)
	  A(0,i) = A(b0,i); // slow, should add call to matrix<>
	
	if(b1 != 1)
	  for(unsigned int i=0;i<A.xsize();i++)
	    A(1,i) = A(b1,i);
      }
      else if(b1 == 0){
	if(b0 != 1){
	  for(unsigned int i=0;i<A.xsize();i++)
	    A(1,i) = A(0,i);
	  
	  for(unsigned int i=0;i<A.xsize();i++)
	    A(0,i) = A(b0,i);
	}
	else{
	  for(unsigned int i=0;i<A.xsize();i++){
	    math::blas_real<float> tmp = A(0,i);
	    A(0,i) = A(1,i);
	    A(1,i) = tmp;
	  }
	}
      }
      
      // rows are now in place, scales both vectors so that
      // variance of dimensions is 1
      
      v0 = 1.0f/::sqrtf(v0);
      v1 = 1.0f/::sqrtf(v1);
      
      for(unsigned int i=0;i<A.xsize();i++){ // slow...
	A(0,i) *= v0;
	A(1,i) *= v1;
      }
      
      A.resize(2,dimension);
      
      // decorrelates and dimension reduces data
      
      for(unsigned int i=0;i<data.size();i++)
	data[i] = A*data[i];
    }
    
    
    
    // starts dlib
    
    dlib.extensions("verbose off");
    dlib.setName("whiteice data visualization");
    
    if(dlib.open(800,600) == false){    
      std::cout << "opening graphics device failed." << std::endl;
      delete font;
      return -1;
    }
    
    font->setBuffer(dlib.width(), dlib.height(), dlib.buffer());
    
    
    dlib.clear(rgblist::WHITE);
    
    // font->print(10, font->height()*2, i->c_str(), rgblist::BLACK);

    show_scatterplot(data, font, labels, labeled);
    
    font->print(10, (int)(font->height()*(3 + 4*1.5)),
		"press any key to continue..", rgblist::BLACK);
    
    dlib.update();
    
    
    while(!dlib.kbhit()){
      struct timespec sp;
      sp.tv_sec  = 0;
      sp.tv_nsec = 1000000;
      nanosleep(&sp, 0);
    }
      
    while(dlib.kbhit()) dlib.key();
    
  }
  else{ // else: normal visualization task
  
    if(plotsymbols.size() == 0){
      std::cout << "no symbols for visualization." << std::endl;
      delete font;
      return -1;
    }
    
    
    // filters unexisting symbolnames away
    for(std::vector<std::string>::iterator i=plotsymbols.begin();
	i!=plotsymbols.end();)
    {
      if(datafile.exists(*i) == false){
	std::cout << "symbol " << *i 
		  << " doesn't exist - ignoring." << std::endl;
	
	plotsymbols.erase(i); // erases this symbol
      }
      else i++;
    }
    
    
    if(plotsymbols.size() == 0){
      delete font;
      return -1; // all var names were fake
    }
    
    // starts dlib
    dlib.extensions("verbose off");
    dlib.setName("whiteice data visualization");
    
    if(dlib.open(800,600) == false){    
      std::cout << "opening graphics device failed." << std::endl;
      delete font;
      return -1;
    }
    
    font->setBuffer(dlib.width(), dlib.height(), dlib.buffer());
    
    
    for(std::vector<std::string>::iterator i=plotsymbols.begin();
	i!=plotsymbols.end();i++)
    {
      dlib.clear(rgblist::WHITE);
      
      font->print(10, font->height()*2, i->c_str(), rgblist::BLACK);
      
      std::vector<int> ivalues;
      std::vector<float> fvalues;
      std::vector<std::string> svalues;
      
      // integers
      if(datafile.get(*i, ivalues) == true){
	show_statistics(ivalues, font);
	show_plot(ivalues, font);
      }
      
      // floats
      if(datafile.get(*i, fvalues) == true){
	show_statistics(fvalues, font);
	show_plot(fvalues, font);
      }
      
      // strings
      if(datafile.get(*i, svalues) == true){
	show_statistics(svalues, font);
	show_plot(svalues, font);
      }
      
      
      font->print(10, (int)(font->height()*(3 + 4*1.5)),
		  "press any key to continue..", rgblist::BLACK);
      
      dlib.update();
      
      
      while(!dlib.kbhit()){
	struct timespec sp;
	sp.tv_sec  = 0;
	sp.tv_nsec = 10000;
	nanosleep(&sp, 0);
      }
      
      while(dlib.kbhit()) dlib.key();
    }
        
  }
  
  delete font;
  return 0;
}



bool init(int argc, char** argv,
	  std::string& filename,
	  std::vector<std::string>& options,
	  std::vector<std::string>& plotsymbols,
	  Font*& font) throw()
{
  srand(time(0));
  
  if(argc < 2) // not enough options
    return false;
  
  
  for(int i=1;i<(argc-1);i++){
    if(strcmp("-l", argv[i]) == 0)
      options.push_back(LIST_SYMBOLS_OPTION);
    else if(strcmp("-s", argv[i]) == 0)
      options.push_back(SCATTERPLOT_OPTION);
    else{
      for(int j=i;j<(argc-1);j++)
	plotsymbols.push_back(argv[j]);
      break;
    }
  }
  
  
  filename = argv[argc-1];
  
  try{
    // TODO: scale size with resolution
    font = new Font("Vera.ttf", 12);
  }
  catch(std::exception& e){
    return false; // no font
  }
  
  return true;
}


void show_usage() throw()
{
  std::cout << "usage: wdv [-l] [-s] [list of symbols] datafilename" << std::endl;
}


bool has_option(std::vector<std::string>& options, std::string opname) throw()
{
  try{
    std::vector<string>::iterator i = options.begin();
    
    while(i != options.end()){
      if(*i == opname) return true;
      i++;
    }
    
    return false;
  }
  catch(std::exception& e){ return false; }
}


void show_statistics(const std::vector<int>& ivalues,
		     Font* font) throw()
{
  std::vector<float> ifl;
  ifl.resize(ivalues.size());
  
  for(unsigned int i=0;i<ivalues.size();i++)
    ifl[i] = ivalues[i];
  
  show_statistics(ifl, font);
}


void show_statistics(const std::vector<float>& fvalues,
		     Font* font) throw()
{
  // calculates mean, variance and third moment
  float ex, exx, exxx;
  ex    = 0.0f;
  exx   = 0.0f;
  exxx  = 0.0f;
  
  for(unsigned int i=0;i<fvalues.size();i++){
    ex   += fvalues[i];
    exx  += fvalues[i]*fvalues[i];
    exxx += fvalues[i]*fvalues[i]*fvalues[i];
  }
  
  ex   /= (float)fvalues.size();
  exx  /= (float)fvalues.size();
  exxx /= (float)fvalues.size();
  
  
  char buf[80];
  
  sprintf(buf, "%d samples", (int)fvalues.size());
  font->print(10, (int)(font->height()*(3 + 1*1.5)),
	      buf, rgblist::BLACK);
  
  sprintf(buf, "mean %+2.2f  var %+2.2f  3rd mom. %+2.2f",
	  ex, exx - ex*ex, exxx);
  font->print(10, (int)(font->height()*(3 + 2*1.5)),
	      buf, rgblist::BLACK);
  
}


void show_statistics(const std::vector<std::string>& svalues,
		     Font* font) throw()
{
  // calculates mean, variance and E[x^3] of string length
  float ex, exx, exxx;
  ex    = 0.0f;
  exx   = 0.0f;
  exxx  = 0.0f;
  
  for(unsigned int i=0;i<svalues.size();i++){
    ex   += svalues[i].size();
    exx  += svalues[i].size()*svalues[i].size();
    exxx += svalues[i].size()*svalues[i].size()*svalues[i].size();
  }
  
  ex   /= (float)svalues.size();
  exx  /= (float)svalues.size();
  exxx /= (float)svalues.size();
  
  
  char buf[80];
  
  sprintf(buf, "%d text strings", (int)svalues.size());
  font->print(10, (int)(font->height()*(3 + 1*1.5)),
	      buf, rgblist::BLACK);
  
  sprintf(buf, "mean %+2.2f  var %+2.2f  3rd mom. %+2.2f",
	  ex, exx - ex*ex, exxx);
  font->print(10, (int)(font->height()*(3 + 2*1.5)),
	      buf, rgblist::BLACK);
}


void show_plot(const std::vector<int>& ivalues,
	       Font* font) throw()
{
  std::vector<float> ifl;
  ifl.resize(ivalues.size());
  
  for(unsigned int i=0;i<ivalues.size();i++)
    ifl[i] = ivalues[i];
  
  show_plot(ifl, font);
}



/*
 * FIXME: this is ugly coded hack
 */
void show_plot(const std::vector<float>& fvalues,
	       Font* font) throw()
{
  // finds min, max, mean, var
  // plots traditional graphical data
  
  float ef,fmin,fmax, var;
  ef   = 0.0f;
  fmin =  10000000000000000000000.0f;
  fmax = -10000000000000000000000.0f;
  var  = 0.0f;
  
  for(std::vector<float>::const_iterator i=fvalues.begin();
      i!=fvalues.end();i++)
  {
    ef += *i;
    var += (*i)*(*i);
    if(*i > fmax) fmax = *i;
    if(*i < fmin) fmin = *i;    
  }
  
  ef /= (float)fvalues.size();
  var /= (float)fvalues.size();
  var -= ef*ef;
  
  // calculates origo and scaling for graph
  float ox, oy; // origo
  float sx, sy; // scaling
  
  sx = (float)dlib.width() / (float)fvalues.size();
  
  if(2.0f*var < (fmax - fmin)/2.0f){
    // the are some rare max/min values -> use variance
    sy = dlib.height() / (float)(2.0f*var);
  }
  else{
    // using variance would clip much data -> use min/max
    sy = dlib.height() / (float)(fmax - fmin);
  }
  
  ox = dlib.width()/2.0f;
  oy = dlib.height()/2.0f;
  
  unsigned int old_x, old_y;
  old_x = 0;
  old_y = (int)oy;
  
  // plots axes
  {
    char buf[80];
    
    line(0, (int)oy, dlib.width()-1, (int)oy, rgblist::BLACK);
    line((int)ox, 0,
	 (int)ox, dlib.height()-1, rgblist::BLACK);        
    
    // small ticks
    
    std::vector<unsigned int> xticks;
    std::vector<int> xvalues;
    
    // x axis
    {
      int index = -((signed)fvalues.size()/2);
      unsigned int counter = 0;
    
      for(std::vector<float>::const_iterator i=fvalues.begin();
	  i!=fvalues.end();i++, index++, counter++)
      {
	if(counter > (fvalues.size()/16))
	  counter = 0;
	else{
	  counter++;
	  continue;
	}

	if(index == 0)
	  continue;
	
	int x = (int)(ox + sx*index);
	
	line(x, (int)(oy - dlib.height()*0.007),
	     x, (int)(oy + dlib.height()*0.007), rgblist::BLACK);
	
	// (int)(ox - dlib.width()*0.007), i,
	
	// prints numbers
	sprintf(buf, "%d", index);
	
	if(index <= 0){
	  font->print(x - (int)(font->width('_')*1.3),
		      (int)(oy + font->height()*1.8),
		      buf, rgblist::BLACK);
	}
	else{
	  font->print(x - (int)(font->width('_')*1.1),
		      (int)(oy + font->height()*1.8),
		      buf, rgblist::BLACK);
	}
	
      }
    }
    
    
    // y axis
    {
      for(int i=0;i<dlib.height();i+=(dlib.height()/12)){
	line((int)(ox - dlib.width()*0.007), i,
	     (int)(ox + dlib.width()*0.007), i, rgblist::BLACK);
	
	float fy = (dlib.height() - i - oy)/sy;
	
	if(fabs(fy) < ((1/sy)/1000.0f))
	  continue;
	
	// prints numbers
	sprintf(buf, "%.2f", fy);
	
	font->print((int)(ox + dlib.width()*0.01),
		    i + (int)((font->height()*1.2)),
		    buf, rgblist::BLACK);
      }
    }
    
  }
  
  // plots data
  
  int index = -((signed)fvalues.size()/2);
  
  for(std::vector<float>::const_iterator i=fvalues.begin();
      i!=fvalues.end();i++, index++)
  {
    int x = (int)(ox + sx*index);
    int y = dlib.height() - (int)(oy + sy*(*i));
    
    if(x < 0 || x >= dlib.width() || 
       y < 0 || y >= dlib.height())
      continue; // out of screen area data
    
    line(old_x, old_y, x, y, rgblist::BLUE);
    
    old_x = x;
    old_y = y;
  }
  
}


void show_plot(const std::vector<std::string>& svalues,
	       Font* font) throw()
{
  std::cout << "no string visualization" << std::endl;
}


void show_scatterplot(const std::vector< math::vertex< math::blas_real<float> > >& data, Font* font,
		      const std::vector<std::string>& labels, const std::vector<unsigned int>& labeled)
{
  if(font == 0 || data.size() == 0)
    return;
  
  if(data[0].size() != 2)
    return;
  
  // calculates origo and scaling for graph  
  
  math::vertex< float > origo(2);
  float sx, sy; // scaling
  
  origo[0] = dlib.width()/2.0f;
  origo[1] = dlib.height()/2.0f;
  
  sx = dlib.width()/6.0f;
  sy = dlib.height()/6.0f;
  
  
  // plots axes
  {
    char buf[80];
    
    line(0, (int)origo[1], dlib.width()-1, (int)origo[1]  , rgblist::BLACK);
    line((int)origo[0], 0, (int)origo[0] , dlib.height()-1, rgblist::BLACK);
    
    // small ticks
    
    std::vector<unsigned int> xticks;
    std::vector<int> xvalues;
    
    // x axis
    {
      for(int i=(dlib.width() % 12)/2;i<dlib.width();i+=(dlib.width()/12)){
	line(i, (int)(origo[1] - 0.007f*dlib.height()),
	     i, (int)(origo[1] + 0.007f*dlib.height()), rgblist::BLACK);

	float fx = (i - origo[0])/sx;
	
	if(fabs(fx) < ((1/sx)/1000.0f))
	  continue;
	
	// prints numbers
	sprintf(buf, "%.1f0", fx  );

	font->print(i - (int)(font->width('_')*1.5),
		    (int)(origo[1] + font->height()*1.8),
		    buf, rgblist::BLACK);

      }
    }
    
    
    // y axis
    {
      for(int i=0;i<dlib.height();i+=(dlib.height()/12)){
	line((int)(origo[0] - dlib.width()*0.007), i,
	     (int)(origo[0] + dlib.width()*0.007), i, rgblist::BLACK);
	
	float fy = (dlib.height() - i - origo[1])/sy;
	
	if(fabs(fy) < ((1/sy)/1000.0f))
	  continue;
	
	// prints numbers
	sprintf(buf, "%.1f0", fy);
	
	font->print((int)(origo[0] + dlib.width()*0.01),
		    i + (int)((font->height()*1.2)),
		    buf, rgblist::BLACK);
      }
    }
    
  }
  
  // plots data
  
  int radius = 3*((dlib.width()*dlib.height())/((800*600)));
  
  for(unsigned int i=0;i<data.size();i++){
    int x = (int)(sx*data[i][0].value() + origo[0]);
    int y = (int)(sy*data[i][1].value() + origo[1]);
    
    circle(x, y, radius, rgblist::BLUE);
  }
  
  // shows labels
  
  for(unsigned int i=0;i<labeled.size();i++){
    int x = (int)(sx*data[labeled[i]][0].value() + origo[0]);
    int y = (int)(sy*data[labeled[i]][1].value() + origo[1]);
    
    font->print(x, y, labels[i].c_str(), rgblist::BLACK);
  }
    
  
}

