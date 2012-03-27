#ifndef SPHERE_H
#define SPHERE_H

#include "Vector.h"
#include "Ray.h"
#include "Volume.h"

class Sphere : public Volume
{
public:
    Sphere();
    Sphere(const int etype);
    Sphere(Sphere& other);
    Sphere(double x, double y, double z, double rad);
    ~Sphere();

	LINEAGE2("Sphere", "Volume")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	
	
	void setPos(Vector* p);
	Vector* pos();
	
	void setRadius(double r);
	double radius() const;

// 	void draw(const DrawOptions* drawopts, const CameraGL* cam);
	bool rayIntersect(const Ray& ray, double& t);
	
	virtual bool contains(const Vector& v, double expand=0.0);

	bool overlapRadius(Sphere& s2, double m=1.0);
	bool overlapRadius(Sphere* s2, double m=1.0);

	double volume();
	
private:
	virtual void updateBoundingBox();
	Vector* _pos;
	double _radius;
};

#endif
