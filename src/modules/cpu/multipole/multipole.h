#include <vector>
using namespace std;

class Multipole
{
public:
	Multipole(const int count);
	void zero();
	vector<double> values;
	int count;
	int l;
};

class MultipoleCartesian : public Multipole
{
public:
	MultipoleCartesian(const int _lmax = 5);

/*
	void SetToVector(const MVector &); 
	//set values = {0, c[0], c[1], c[2], 0, 0, 0, ..}
	void SetToDeriv(const MVector &);//set values to D[r] see paper
	//set values_p = Sum_t{W(t)*(-.5*t)^p/p!}
	void AddTaylor(const mpole *, const MVector &);//add Taylor expansion
	void AddUpConv(const mpole *, const mpole *);//up-convolution
	void AddDownConv(const mpole *, const mpole *);//down-convolution
	void AddShift(const mpole*, const MVector &);//shift the origin of multipole moment
	MVector average() const;  //returns average field, used in CalcField for smallest cell
*/	
};


class MultipoleSphericalHarmonics : public Multipole
{
public:
	MultipoleSphericalHarmonics(const int _lmax = 3);

};


