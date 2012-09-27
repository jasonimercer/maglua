#include "Vector_stuff.h"
#include <complex>
using namespace std;

int do_wood_calculation(Cvctr& H, Cvctr& M, Cvctr& K, Cvctr& Mout);
int do_wood_calculation_demag_min_close(Cvctr& H, Cvctr& M, Cvctr& K, Cvctr& M0, double DN);
int do_wood_calculation_demag_min_far(Cvctr& H, Cvctr& M, Cvctr& K, Cvctr& M0, double DN);
int do_wood_calculation_demag_max(Cvctr& H, Cvctr& M, Cvctr& K, Cvctr& M0, double DN);
bool Stability(double mu, double mq, double hu, double hq);
complex<double> ACOS_COMP( complex<double> Z);
int EffAni(Cvctr& K, Cvctr M, double Nx, double Ny, double Nz);
int EffAni2(Cvctr& K, Cvctr M, double Nx, double Ny, double Nz);
double DU(Cvctr H, Cvctr K, Cvctr M, double DN);

