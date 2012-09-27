#ifndef VECTOR_STUFF_H
#define VECTOR_STUFF_H


class Cvctr {
	public:
		double x, y, z,mag;
    void set_cmp (double,double,double);
	void set_ang (double,double,double);
	void smult(double);
	double DELTA(Cvctr);
	Cvctr uVect();
	Cvctr vadd(Cvctr);
	Cvctr vsub(Cvctr);
	double dot(Cvctr);
	Cvctr cross(Cvctr);
	//JM edit, added a constructor with optional values 
	//and a copy constructor incase you want to use this
	//object with the standard template library
	Cvctr(double x=0, double y=0, double z=0); 
	Cvctr(const Cvctr& other); 
};

#endif
