/*
 * generic 2d and 3d point classes
 *
 * DEPRECATED: use vertex instead
 */
#ifndef point_h
#define point_h

#include "printable.h"

namespace whiteice
{
  
  template <typename T>
    class point2d : public printable
    {
    public:
      
      point2d(const T& x = 0, const T& y = 0);
      virtual ~point2d();
      
      const T& getX() const;
      const T& getY() const;
      T& getX();
      T& getY();
      
      bool setX(const T& x);
      bool setY(const T& y);
      
      point2d<T>  operator+ (const point2d<T>& p) const;
      point2d<T>  operator- (const point2d<T>& p) const;
      T   operator* (const point2d<T>& p) const;
      point2d<T>& operator+=(const point2d<T>& p);
      point2d<T>& operator-=(const point2d<T>& p);
      
      T& operator[](const unsigned int);
      const T& operator[](const unsigned int) const;
      
      
      bool print();
      
    private:
      
      T x;
      T y;
      
    };
  
  template <typename T>
    class point3d : public printable
    {
    public:
      
      point3d(const T& x = 0, const T& y = 0, const T& z = 0);
      virtual ~point3d();
      
      const T& getX() const;
      const T& getY() const;
      const T& getZ() const;
      T& getX();
      T& getY();
      T& getZ();
      
      
      bool setX(const T& x);
      bool setY(const T& y);
      bool setZ(const T& z);
      
      point3d<T>  operator+ (const point3d<T>& p) const;
      point3d<T>  operator- (const point3d<T>& p) const;
      T   operator* (const point3d<T>& p) const;
      point3d<T>& operator+=(const point3d<T>& p);
      point3d<T>& operator-=(const point3d<T>& p);
      
      T& operator[](const unsigned int);
      const T& operator[](const unsigned int) const;
      
      
      bool print();
      
    private:
      
      T x,y,z;
    };

}
  
#include "point.cpp"
  
#endif


