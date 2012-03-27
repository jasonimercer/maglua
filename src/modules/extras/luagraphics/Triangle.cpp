#include "Triangle.h"
#include <math.h>

Triangle::Triangle()
{
}

double Triangle::area()
{
	const Vector& a = vert[0];
	const Vector& b = vert[1];
	const Vector& c = vert[2];
	return 0.5 * Vector::crossProduct(b-a, c-a).length();	
}

Triangle::Triangle(const Vector& a, const Vector& b, const Vector& c, const Vector& backside)
{
	vert[0] = a;
	vert[1] = b;
	vert[2] = c;
	
	bb.include(a);
	bb.include(b);
	bb.include(c);
	
	bad = true; //will get changed to false in calcPlane
	if(area() > 1E-8)
		calcPlane(backside);
}

Triangle::Triangle(const Vector& a, const Vector& b, const Vector& c)
{
	vert[0] = a;
	vert[1] = b;
	vert[2] = c;
		
	bb.include(a);
	bb.include(b);
	bb.include(c);
	
	bad = true; //will get changed to false in calcPlane
	if(area() > 1E-8)
	{
		//calcPlane(0.3333*(a+b+c) - Vector::crossProduct(b-a, c-a));
		calcPlane(Vector(0,0,0));
	}
}

double Triangle::maxEdgeLength()
{
	double m = 0;
	
	for(int i=0; i<3; i++)
	{
		double M = (vert[i] - vert[(i+1)%3]).length();
		if(M > m)
			m = M;
	}
	return m;
}


// void Triangle::drawVerts() const
// {
// 	glNormal3fv(plane);
// 	//	static const double uv[3][2] = {{0,0}, {1,1}, {1,0}};
// 	for(int i=0; i<3; i++)
// 	{
// 		//		glTexCoord2fv(uv[i]);
// 		glVertex3f(vert[i].x(), vert[i].y(), vert[i].z());
// 	}
// }
// 
// void Triangle::drawWireframe() const
// {
// 	glBegin(GL_LINE_LOOP);
// 	drawVerts();
// 	glEnd();
// }

void Triangle::flipNormal()
{
// 	for(int i=0; i<4; i++)
// 		plane[i] *= -1.0;
	normal *= -1.0;
	planeOffset *= -1.0;
}

void Triangle::calcNormal()
{
	normal = Vector::crossProduct(vert[1] - vert[0], vert[2] - vert[0]).normalized();
}

void Triangle::calcPlane(const Vector& backside, int rec)
{
	if(rec > 3)
	{
		bad = true;
		return;
	}
	
	calcNormal();
	planeOffset = -1.0 * (Vector::dotProduct(vert[0], normal));
	
	// test backside
	// first test for coplanar
	double v = value(backside);
	bad = fabs(v) < 1E-14;
	if(bad)
		return;
	
	// if v > 0 then point infront, swap verts and try again
	if(v > 0)
	{
		vert[1].swap(vert[2]);
		calcPlane(backside, rec+1);
	}
}

double Triangle::value(const Vector& point) const
{
	return normal.dot(point) + planeOffset;
}

bool Triangle::infront(const Vector& point) const
{
	return value(point) > 0;
}

double Triangle::distancePointPlane(const Vector& point) const
{
	return fabs(value(point));
}

bool Triangle::sphereIsect(const Sphere& s) const
{
	return sphereIsect(s.pos(), s.radius());
}
bool Triangle::sphereIsect(const Vector& pp, const double radius) const
{
	const Vector& center = pp;
	const double rad = radius;

	double distance = (distancePointPlane(center));
	if( fabs(distance) > rad)
		return false;

// 	if(bb.excludes(pp,radius))
// 		return false;

	//project center of sphere onto plane
	Vector p(center + distance * normal);
	double rp2 = rad*rad - distance*distance; //radius squared of circle on plane
	
	//check to see if verts of triangle are inside projected circle
	for(int i=0; i<3; i++)
	{
		double d2 = Vector::dotProduct(vert[i] - p, vert[i] - p);
		if(d2 < rp2)
			return true;
	}
	
	//check to see if circle crosses the edges
	for(int i=0; i<3; i++)
	{
		int a =  i;
		int b = (i+1)%3;
		
		const Vector& x1 = vert[a];
		const Vector& x2 = vert[b];
		const Vector&  c = p;
		
		// We will solve for t in line segment = x1 + (x2-x1)t
		// solve for places where the line and circle intersect
		//
		// t varies from 0 to 1
		//
		//
		// solution for t is:
		//
		//     (x1.x1+c.x2-c.x1-x1.x2) +- sqrt[ ((c-x1).(x1-x2))^2 - (c.c - r^2 - 2 c.x1+x1.x1)((x1-x2).(x1-x2))]
		// t = --------------------------------------------------------------------------------------------------
		//                                       ((x1-x2).(x1-x2))
		
		#define dot(a,b) Vector::dotProduct((a),(b))
		const double n1 = dot(x1,x1)+dot(c,x2)-dot(c,x1)-dot(x1,x2);
		const double n2 = dot(c-x1, x1-x2);
		const double n3 = dot(c,c) - rp2 - 2*dot(c,x1)+dot(x1,x1);
		const double n4 = dot(x1-x2,x1-x2);
		const double n5 = n4;
		#undef dot
		
		// if solution is not complex and one of them are between 0
		// and 1, the projected circle intersects the line segment
		if(n2*n2-n3*n4 >= 0)
		{
			double t0 = (n1 + sqrt(n2*n2-n3*n4)) / n5;
			double t1 = (n1 - sqrt(n2*n2-n3*n4)) / n5;
			
			if(t0 >= 0 && t0 <= 1)
				return true;
			if(t1 >= 0 && t1 <= 1)
				return true;
		}
	}
	
	//finally, check to see if circle center is inside triangle
	Vector v1(Vector::crossProduct(p-vert[1], vert[2]-vert[1]));
	Vector v2(Vector::crossProduct(p-vert[2], vert[0]-vert[2]));
	Vector v3(Vector::crossProduct(p-vert[0], vert[1]-vert[0]));
	
	//if these 3 vectors are all in the same direction then the point is in the triangle
	if(Vector::dotProduct(v1, v2) > 0 && Vector::dotProduct(v1, v3) > 0)
		return true;
	
	
	return false;
}
