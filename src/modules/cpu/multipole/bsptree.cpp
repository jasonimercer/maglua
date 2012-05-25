#include "bsptree.h"
#include <math.h>

BSPTree::BSPTree(dArray* X, dArray* Y, dArray* Z, dArray* Weight, BSPTree* Parent)
{
	x = luaT_inc<dArray>(X);
	y = luaT_inc<dArray>(Y);
	z = luaT_inc<dArray>(Z);
	weight = luaT_inc<dArray>(Weight);
	
	c[0] = 0;
	c[1] = 0;
	
	parent = Parent;
	if(parent)
	{
		split_dir = (parent->split_dir + 1) % 3;
	}
	else //this is the root and should have all members
	{
		for(int i=0; i<x->nxyz; i++)
			members.push_back(i);
	}
}


BSPTree::~BSPTree()
{
	if(c[0])
	{
		delete c[0];
		delete c[1];
	}

	luaT_dec<dArray>(x);
	luaT_dec<dArray>(y);
	luaT_dec<dArray>(z);
}


void BSPTree::split(int until_contains)
{
	if(c[0]) return; //already split
	
	if((int)members.size() <= until_contains)
		return; //no need to split more
	
	double u[3];
	double s[3];
	getStats(u, s);

	dArray* a[3] = {x,y,z};
	
	// splitting on split_dir
	split_value = u[split_dir];
	
	c[0] = new BSPTree(x, y, z, weight, this);
	c[1] = new BSPTree(x, y, z, weight, this);
	
	for(unsigned int i=0; i<members.size(); i++)
	{
		int j = members[i];
		
		if( (*a[split_dir])[j] < split_value )
			c[0]->members.push_back(j);
		else
			c[1]->members.push_back(j);
	}
	
	if(until_contains > 0)
	{
		c[0]->split(until_contains);
		c[1]->split(until_contains);
	}
}

	
void BSPTree::getStats(double* meanXYZ, double* stddevXYZ)
{
	  meanXYZ[0] = 0;
	  meanXYZ[1] = 0;
	  meanXYZ[2] = 0;
	stddevXYZ[0] = 0;
	stddevXYZ[1] = 0;
	stddevXYZ[2] = 0;
	
	if(!x || members.size() == 0)
		return;
	
	double sum_weight = 0;
	
	for(unsigned int i=0; i<members.size(); i++)
	{
		int j=members[i];
		double w = (*weight)[j];
		meanXYZ[0] += (*x)[j] * w;
		meanXYZ[1] += (*y)[j] * w;
		meanXYZ[2] += (*z)[j] * w;
		sum_weight += w;
	}
	if(sum_weight == 0)
		sum_weight = 1.0;
	meanXYZ[0] /= sum_weight;
	meanXYZ[1] /= sum_weight;
	meanXYZ[2] /= sum_weight;
	
	for(unsigned int i=0; i<members.size(); i++)
	{
		int j=members[i];
		double w = (*weight)[j];
		stddevXYZ[0] += pow((*x)[j]*w - meanXYZ[0], 2);
		stddevXYZ[1] += pow((*y)[j]*w - meanXYZ[1], 2);
		stddevXYZ[2] += pow((*z)[j]*w - meanXYZ[2], 2);
	}
	
	stddevXYZ[0] /= sum_weight;
	stddevXYZ[1] /= sum_weight;
	stddevXYZ[2] /= sum_weight;
	
	stddevXYZ[0] = sqrt(stddevXYZ[0]);
	stddevXYZ[1] = sqrt(stddevXYZ[1]);
	stddevXYZ[2] = sqrt(stddevXYZ[2]);
}













