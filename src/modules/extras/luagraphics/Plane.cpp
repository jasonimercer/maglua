#include "Plane.h"

Plane::Plane(const Vector& n, const Vector& x0)
{
	normal = n.normalized();
	d = - normal.dot(x0);
}


bool Plane::rayIntersect(const Ray& ray, double& t)
{
	// ax + by + cz + d = 0
	// n = (a, b, c)
	
	// r = o + t d
	
	//if ray.direction dot plane.normal == 0 then no intersection
	if(ray.direction->dot(normal) == 0)
		return false;
	
	t = (-d - normal.dot(*ray.origin)) / (normal.dot(*ray.direction));
	return true;
}
