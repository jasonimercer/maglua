#include "Tetrahedron.h"
#include <math.h>

inline double tetvol(const Vector& a, const Vector& b, const Vector& c, const Vector& d)
{
	//V = 1/3! | a . (b x c)|
	Vector ea(b-a);
	Vector eb(c-a);
	Vector ec(d-a);
	
	return (1.0/6.0) * fabs(Vector::dotProduct(ea, Vector::crossProduct(eb, ec)));
}

Tetrahedron::Tetrahedron(const Vector& a, const Vector& b, const Vector& c, const Vector& d)
{
	init(a, b, c, d);
}

Tetrahedron::Tetrahedron(const Triangle& tri, const Vector& d)
{
	init(tri.vert[0], tri.vert[1], tri.vert[2], d);
}

Tetrahedron::~Tetrahedron()
{
}

bool Tetrahedron::rayIntersect(const Ray& ray, double& t)
{
	
}

void Tetrahedron::updateBoundingBox()
{
	bb.reset();
	for(int i=0; i<4; i++)
	{
		bb.include(vert[i]);
	}
}


double Tetrahedron::maxEdgeLength()
{
	double m = 0;

	for(int i=0; i<4; i++)
	{
		double M = tri[i].maxEdgeLength();
		if(M > m)
			m = M;
	}
	return m;
}

bool Tetrahedron::sphereIsect(const Sphere& s) const
{
	const Vector& center = s.pos();
	const double rad = s.radius();
	
	//check for bounding box first
	if(center.x() + rad < aabb[0]) return false;
	if(center.x() - rad > aabb[1]) return false;
	if(center.y() + rad < aabb[2]) return false;
	if(center.y() - rad > aabb[3]) return false;
	if(center.z() + rad < aabb[4]) return false;
	if(center.z() - rad > aabb[5]) return false;

	//check if sphere intersects with each triangle
	for(int i=0; i<4; i++)
	{
		if(tri[i].sphereIsect(s))
			return true;
	}

	//check if sphere center is inside tetrahedron
	for(int i=0; i<4; i++)
	{
		if(tri[i].infront(center))
			return false; // must be behind all 4 faces to be inside
	}
	return true;
}

bool Tetrahedron::contains(const Vector& p, double expand)
{
	for(int i=0; i<4; i++)
	{
		if(tri[i].value(p) > -expand)
			return false;
	}
	return true;
}

bool Tetrahedron::excludes(const Vector& p, double expand)
{
// 	if(vol == 0)
// 	{
// // 		printf("vol == 0\n");
// 		return true;
// 	}
	
	if(bb.excludes(p,expand))
	{
// 		printf("bb excludes\n");
		return true;
	}
	
	if(contains(p,expand))
		return false;
	
	for(int i=0; i<4; i++)
	{
		if(tri[i].sphereIsect(p, expand))
		{
			return false;
		}
	}

// 	printf("fall through excludes\n");
	return true;
}

void Tetrahedron::init(const Vector& a, const Vector& b, const Vector& c, const Vector& d)
{
	bad = false;
	vert[0] = a;
	vert[1] = b;
	vert[2] = c;
	vert[3] = d;

	bb.reset();
	
	
	for(int i=0; i<4; i++)
	{
		bb.include(vert[i]);
		neighbours[i] = 0;
	}

	vol = tetvol(vert[0], vert[1], vert[2], vert[3]);


	if(vol < 1E-10)
	{
		//make it really ill-defined
		vert[0] = a;
		vert[1] = a;
		vert[2] = a;
		vert[3] = a;
		bb.reset();
		bad = true;
		return;
	}

	aabb[0] = vert[0].x();
	aabb[1] = vert[0].x();
	aabb[2] = vert[0].y();
	aabb[3] = vert[0].y();
	aabb[4] = vert[0].z();
	aabb[5] = vert[0].z();

	for(int i=0; i<4; i++)
	{
		if(vert[i].x() < aabb[0])
			aabb[0] = vert[i].x();
		if(vert[i].x() > aabb[1])
			aabb[1] = vert[i].x();

		if(vert[i].y() < aabb[2])
			aabb[2] = vert[i].y();
		if(vert[i].y() > aabb[3])
			aabb[3] = vert[i].y();

		if(vert[i].z() < aabb[4])
			aabb[4] = vert[i].z();
		if(vert[i].z() > aabb[5])
			aabb[5] = vert[i].z();
	}


	tri[0].vert[0] = a;
	tri[0].vert[1] = b;
	tri[0].vert[2] = c;

	tri[1].vert[0] = b;
	tri[1].vert[1] = a;
	tri[1].vert[2] = d;

	tri[2].vert[0] = c;
	tri[2].vert[1] = b;
	tri[2].vert[2] = d;

	tri[3].vert[0] = a;
	tri[3].vert[1] = c;
	tri[3].vert[2] = d;

	Vector m(0,0,0); //middle
	for(int i=0; i<4; i++)
	{
		m += vert[i];
	}

	m *= 0.25;

	for(int i=0; i<4; i++)
	{
		tri[i].calcPlane(m);

		if(tri[i].bad)
		{
// 			cout << i << endl;
// 			cout << m << endl;
// 			cout << tri[i].value(m) << endl;
// 			for(int j=0; j<4; j++)
// 				cout << " (" << tri[j].vert[0] << ":"<< tri[j].vert[1] << ":"<< tri[j].vert[2] << ")" << endl;
// 			cout << endl;
			
			bad = true;
			return;
		}
	}

}

double Tetrahedron::volume()
{
	return vol;
}


