#ifndef PLANE_H
#define PLANE_H

#include "Vector.h"
#include "Ray.h"

#include "libLuaGraphics.h"
class LUAGRAPHICS_API Plane
{
public:
	Plane(const Vector& n=Vector(0,0,1), const Vector& x0=Vector(0,0,0));
	bool rayIntersect(const Ray& ray, double& t);
 
	//Plane& operator =(const Plane& rhs);

	
private:
	Vector normal;
	double d;
};

#endif
