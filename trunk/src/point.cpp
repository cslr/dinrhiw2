/*
 * DEPRECATED: use vertex instead
 *
 */
#ifndef point_cpp
#define point_cpp

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "point.h"

namespace whiteice
{
  
  template <typename T>
  point2d<T>::point2d(const T& x, const T& y)
  {
    this->x = x;
    this->y = y;
  }
  
  
  template <typename T>
  point2d<T>::~point2d(){ }
  
  
  template <typename T>
  const T& point2d<T>::getX() const
  {
    return this->x;
  }
  
  
  template <typename T>
  const T& point2d<T>::getY() const
  {
    return this->y;
  }
  
  
  template <typename T>
  T& point2d<T>::getX()
  {
    return this->x;
  }

  
  template <typename T>
  T& point2d<T>::getY()
  {
    return this->y;
  }
  
  
  template <typename T>
  bool point2d<T>::setX(const T& x)
  {
    this->x = x;
    return true;
  }
  
  
  template <typename T>
  bool point2d<T>::setY(const T& y)
  {
    this->y = y;
    return true;
  }
  
  
  template <typename T>
  bool point2d<T>::print()
  {
    printf("2d point (%f,%f)\n", (float)this->getX(), (float)this->getY());
    return true;
  }
  
  
  template <typename T>
  point2d<T> point2d<T>::operator+ (const point2d<T>& p) const
  {
    point2d<T> q;
  
    q.setX(this->x + p.getX());
    q.setX(this->y + p.getY());
    
    return q;
  }
  
  template <typename T>
  point2d<T> point2d<T>::operator- (const point2d<T>& p) const
  {
    point2d<T> q;
    
    q.setX(this->x - p.getX());
    q.setY(this->y - p.getY());
    
    return q;
  }
  
  template <typename T>
  T point2d<T>::operator* (const point2d<T>& p) const
  {
    T r = this->x * p.getX() + this->y * p.getY();
    return r;
  }
  
  template <typename T>
  point2d<T>& point2d<T>::operator+=(const point2d<T>& p)
  {
    this->x += p.getX();
    this->y += p.getY();
    
    return *this;
  }
  
  
  template <typename T>
  point2d<T>& point2d<T>::operator-=(const point2d<T>& p)
  {
    this->x -= p.getX();
    this->y -= p.getY();
    
    return *this;
  }
  
  
  
  template <typename T>
  T& point2d<T>::operator[](const unsigned int n)
  {
    /* exceptions */
    if(n == 0) return this->x;
    else if(n == 1) return this->y;
    else assert(0);

    // shouldn't ever get here but some compilers complain
    // about missing return value
    return this->x;
  }
  
  template <typename T>
  const T& point2d<T>::operator[](const unsigned int n) const
  {
    /* exceptions */
    if(n == 0) return this->x;
    else if(n == 1) return this->y;
    else assert(0);

    // shouldn't ever get here but some compilers complain
    // about missing return value
    return this->x;
  }
  
  

  
  template <typename T>
  point3d<T>::point3d(const T& x, const T& y, const T& z)
  {
    this->x = x;
    this->y = y;
    this->z = z;
  }
  
  
  template <typename T>
  point3d<T>::~point3d(){ }
  
  
  template <typename T>
  const T& point3d<T>::getX() const
  {
    return this->x;
  }
  
  
  template <typename T>
  const T& point3d<T>::getY() const
  {
    return this->y;
  }
  
  
  template <typename T>
  const T& point3d<T>::getZ() const
  {
    return this->z;
  }
  
  
  template <typename T>
  T& point3d<T>::getX()
  {
    return this->x;
  }

  
  template <typename T>
  T& point3d<T>::getY()
  {
    return this->y;
  }
  
  
  template <typename T>
  T& point3d<T>::getZ()
  {
    return this->z;
  }
  
  
  
  template <typename T>
  bool point3d<T>::setX(const T& x)
  {
    this->x = x;
    return true;
  }
  
  
  template <typename T>
  bool point3d<T>::setY(const T& y)
  {
    this->y = y;
    return true;
  }
  

  template <typename T>
  bool point3d<T>::setZ(const T& z)
  {
    this->z = z;
    return true;
  }
  
  template <typename T>
  bool point3d<T>::print()
  {
    printf("3d point (%f,%f,%f)\n", 
	   (float)this->getX(), (float)this->getY(), (float)this->getZ());
    return true;
  }
  
  
  template <typename T>
  point3d<T> point3d<T>::operator+ (const point3d<T>& p) const
  {
    point3d<T> q;
    
    q.setX(this->x + p.getX());
    q.setY(this->y + p.getY());
    q.setZ(this->z + p.getZ());
    
    return q;
  }
  
  
  template <typename T>
  point3d<T> point3d<T>::operator- (const point3d<T>& p) const
  {
    point3d<T> q;
    
    q.setX(this->x - p.getX());
    q.setY(this->y - p.getY());
    q.setZ(this->z - p.getZ());
    
    return q;
  }
  
  template <typename T>
  T point3d<T>::operator* (const point3d<T>& p) const
  {
    T r = this->x * p.getX() + this->y * p.getY();
    return r;
  }
  
  template <typename T>
  point3d<T>& point3d<T>::operator+=(const point3d<T>& p)
  {
    this->x += p.getX();
    this->y += p.getY();
    this->z += p.getZ();
    
    return *this;
  }
  
  template <typename T>
  point3d<T>& point3d<T>::operator-=(const point3d<T>& p)
  {
    this->x -= p.getX();
    this->y -= p.getY();
    this->z -= p.getZ();
    
    return *this;
  }

  
  template <typename T>
  T& point3d<T>::operator[](const unsigned int n)
  {
    /* exceptions */
    if(n == 0) return this->x;
    else if(n == 1) return this->y;
    else if(n == 2) return this->z;
    else assert(0);
    
    // shouldn't ever get here but some compilers complain
    // about missing return value
    return this->x;
  }
  
  template <typename T>
  const T& point3d<T>::operator[](const unsigned int n) const
  {
    /* exceptions */
    if(n == 0) return this->x;
    else if(n == 1) return this->y;
    else if(n == 2) return this->z;
    else assert(0);

    // shouldn't ever get here but some compilers complain
    // about missing return value
    return this->x;
  }
  
}

#endif
  
