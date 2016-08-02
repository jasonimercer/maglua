#ifndef TETRAHEDRON_H
#define TETRAHEDRON_H

#include "Volume.h"
#include "Vector.h"
#include "Triangle.h"
#include "Sphere.h"

class Tetrahedron : public Volume
{
public:
	Tetrahedron() {}
	Tetrahedron(const Vector& a, const Vector& b, const Vector& c, const Vector& d);
	Tetrahedron(const Triangle& tri, const Vector& d);
	~Tetrahedron();

	void init(const Vector& a, const Vector& b, const Vector& c, const Vector& d);
	bool sphereIsect(const Sphere& s) const;

// 	void draw() const;
// 	void drawCap() const;
// 	void drawWireframe() const;

	bool contains(const Vector& p, double expand=1.0);
	bool excludes(const Vector& p, double expand=1.0);
	double maxEdgeLength();
	Vector vert[4];
	Triangle tri[4];

	double volume();
	bool bad;

	bool rayIntersect(const Ray& ray, double& t);
	
	
	Tetrahedron* neighbours[4];
private:
	virtual void updateBoundingBox();

	double vol;
	double aabb[6];
};



#endif