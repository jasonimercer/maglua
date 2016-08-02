#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "Vector.h"
#include "Sphere.h"
#include "AABB.h"

class Triangle
{
	public:
		Triangle();
		Triangle(const Vector& a, const Vector& b, const Vector& c, const Vector& backside);
		Triangle(const Vector& a, const Vector& b, const Vector& c);
		
		void calcPlane(const Vector& backside, int rec=0);
		void calcNormal();
		void flipNormal();
		double area();
		
		// 	void drawWireframe() const;
		// 	void drawVerts() const;
		
		Vector vert[3];
		//Vector normal;
		//double plane[4];
		Vector normal;
		double planeOffset;
		bool bad;
		
		bool infront(const Vector& point) const;
		double value(const Vector& point) const;
		bool sphereIsect(const Sphere& s) const;
		bool sphereIsect(const Vector& p, const double radius) const;
		double distancePointPlane(const Vector& point) const;
		double maxEdgeLength();
		
		AABB bb;
};

#endif
