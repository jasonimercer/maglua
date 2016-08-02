// Vector_stuff.cpp : Defines the entry point for the console application.
//

#include "Vector_stuff.h"
#include <iostream>
#include <complex>
#include <fstream>
#include <math.h>
using namespace std;

// class Cvctr {
// 	public:
// 		double x, y, z,mag;
//     void set_cmp (double,double,double);
// 	void set_ang (double,double,double);
// 	void smult(double);
// 	double DELTA(Cvctr);
// 	Cvctr uVect();
// 	Cvctr vadd(Cvctr);
// 	Cvctr vsub(Cvctr);
// 	double dot(Cvctr);
// 	Cvctr cross(Cvctr);
// };

Cvctr::Cvctr(double x, double y, double z) //JM edit, added a constructor
{
	set_cmp(x,y,z);
}

Cvctr::Cvctr(const Cvctr& other) //JM edit, added copy constructor
{
	set_cmp(other.x, other.y, other.z);
}


void Cvctr::set_cmp (double a, double b, double c) // set the x,y, and z components of the vector
{
  x = a;
  y = b;
  z = c;
  mag = x*x + y*y + z*z;
  mag = sqrt(mag);
}

void Cvctr::set_ang(double r, double tht, double phi) // set the magnitude and two spherically polar angles of the vector
{
  x = r*cos(phi)*sin(tht);
  y = r*sin(phi)*sin(tht);
  z = r*cos(tht);
  mag = sqrt(x*x + y*y + z*z);
}

void Cvctr::smult(double S)  //multiply the vector by a scalar S.
{
	set_cmp(S*x, S*y, S*z);
//   x = S*x;
//   y = S*y;
//   z = S*z;
//   mag = sqrt(x*x + y*y + z*z);
}

Cvctr Cvctr::vadd(Cvctr V2) //return a vector that is this vector plus vector V2.
{
// 	double vx,vy,vz;
	Cvctr Vout(x + V2.x, y + V2.y, z + V2.z);
// 	vx = x + V2.x;
// 	vy = y + V2.y;
// 	vz = z + V2.z;
// 	Vout.set_cmp(vx,vy,vz);
	return (Vout);
}

Cvctr Cvctr::vsub(Cvctr V2) //return a a vector that is this vector minus vector V2.
{
	double vx,vy,vz;
	Cvctr Vout;
	vx = x - V2.x;
	vy = y - V2.y;
	vz = z - V2.z;
	Vout.set_cmp(vx,vy,vz);
	return (Vout);
}

double Cvctr::dot(Cvctr V2) // return the dot product of this vector projected onto V2.
{
	double dot;
	dot = x*V2.x + y*V2.y + z*V2.z;
	return(dot);
}

Cvctr Cvctr::cross(Cvctr V2) // return the cross product of this vector crossed into V2.
{
	Cvctr cross;
	double vx,vy,vz;
	vx = y*V2.z - z*V2.y;
	vy = z*V2.x - x*V2.z;
	vz = x*V2.y - y*V2.x;
	cross.set_cmp(vx,vy,vz);
	return (cross);
}

Cvctr Cvctr::uVect() // return a unit vector pointing in the same direction as this vector.
{
	Cvctr uV;
	if (mag == 0) uV.set_cmp(0.0,0.0,1.0);
	else uV.set_cmp(x/mag,y/mag,z/mag);
	return(uV);
}

double Cvctr::DELTA(Cvctr V2) // returns the difference in magnitude of the vector created from this vector minus V2
{
	Cvctr diff;
	diff.set_cmp(x,y,z);
	diff = diff.vsub(V2);
	return(diff.mag);
}
