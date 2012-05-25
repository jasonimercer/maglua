#include "octtree.h"
#include <math.h>

OctTree::OctTree(dArray* X, dArray* Y, dArray* Z, OctTree* Parent)
{
	x = luaT_inc<dArray>(X);
	y = luaT_inc<dArray>(Y);
	z = luaT_inc<dArray>(Z);
	
	bounds_low[0] = 0;
	bounds_low[1] = 0;
	bounds_low[2] = 0;

	bounds_high[0] = 1;
	bounds_high[1] = 1;
	bounds_high[2] = 1;
	
	for(int i=0; i<8; i++)
	{
		c[i] = 0;
	}
	
	parent = Parent;
	if(!parent) //this is the root and should have all members
	{
		for(int i=0; i<x->nxyz; i++)
			members.push_back(i);
	}
}


OctTree::~OctTree()
{
	if(c[0])
	{
		for(int i=0; i<8; i++)
			delete c[i];
	}

	luaT_dec<dArray>(x);
	luaT_dec<dArray>(y);
	luaT_dec<dArray>(z);
}

bool OctTree::contains(double px, double py, double pz)
{
	if(px < bounds_low[0] || px >= bounds_high[0]) return false;
	if(py < bounds_low[1] || py >= bounds_high[1]) return false;
	if(pz < bounds_low[2] || pz >= bounds_high[2]) return false;
	return true;
}

void OctTree::setBounds(double* low, double* high, int childNumber)
{
	int a[3] = {1,2,4};
	for(int i=2; i>=0; i--)
	{
		if(childNumber < a[i])
		{
			bounds_low[i] = low[i];
			bounds_high[i] = low[i] + 0.5*(low[i] + high[i]);
		}
		else
		{
			bounds_low[i] = low[i] + 0.5*(low[i] + high[i]);
			bounds_high[i] = high[i];
		}
		childNumber %= a[i];
	}
}



void OctTree::split(int until_contains)
{
	if(c[0]) return; //already split
	
	if((int)members.size() <= until_contains)
		return; //no need to split more
	

	for(int i=0; i<8; i++)
	{
		c[i] = new OctTree(x, y, z, this);
		c[i]->setBounds(bounds_low, bounds_high, i);
	}

	for(unsigned int i=0; i<members.size(); i++)
	{
		int j = members[i];
		for(int k=0; k<8; k++)
		{
			if(c[i]->contains( (*x)[j],  (*y)[j],  (*z)[j]))
			   c[k]->members.push_back(j);
		}
	}
	
	if(until_contains > 0)
	{
		for(int i=0; i<8; i++)
			c[i]->split(until_contains);
	}
}

	
void OctTree::getStats(double* meanXYZ, double* stddevXYZ)
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
		meanXYZ[0] += (*x)[j];
		meanXYZ[1] += (*y)[j];
		meanXYZ[2] += (*z)[j];
		sum_weight += 1.0;
	}
	if(sum_weight == 0)
		sum_weight = 1.0;
	meanXYZ[0] /= sum_weight;
	meanXYZ[1] /= sum_weight;
	meanXYZ[2] /= sum_weight;
	
	for(unsigned int i=0; i<members.size(); i++)
	{
		int j=members[i];
		stddevXYZ[0] += pow((*x)[j] - meanXYZ[0], 2);
		stddevXYZ[1] += pow((*y)[j] - meanXYZ[1], 2);
		stddevXYZ[2] += pow((*z)[j] - meanXYZ[2], 2);
	}
	
	stddevXYZ[0] /= sum_weight;
	stddevXYZ[1] /= sum_weight;
	stddevXYZ[2] /= sum_weight;
	
	stddevXYZ[0] = sqrt(stddevXYZ[0]);
	stddevXYZ[1] = sqrt(stddevXYZ[1]);
	stddevXYZ[2] = sqrt(stddevXYZ[2]);
}













