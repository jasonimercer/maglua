#include "multipole.h"

Multipole::Multipole(const int _count)
{
	count = _count;
	for(int i=0; i<count; i++)
		values.push_back(0);
}

void Multipole::zero()
{
	for(int i=0; i<count; i++)
		values[i] = 0;
}



MultipoleCartesian::MultipoleCartesian(const int l)
	: Multipole((l+1)*(l+2)*(l+3)/6)
{
}

MultipoleSphericalHarmonics::MultipoleSphericalHarmonics(const int l)
	: Multipole((l+1)*(l+1))
{
}

