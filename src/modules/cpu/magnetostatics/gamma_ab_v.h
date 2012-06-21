#ifndef GAMMA_AB_V_H
#define GAMMA_AB_V_H

#ifdef __cplusplus
extern "C"
{
#endif

double gamma_xx_v(double x, double y, double z, const double* prism);
double gamma_yy_v(double x, double y, double z, const double* prism);
double gamma_zz_v(double x, double y, double z, const double* prism);
double gamma_xy_v(double x, double y, double z, const double* prism);
double gamma_yx_v(double x, double y, double z, const double* prism);
double gamma_xz_v(double x, double y, double z, const double* prism);
double gamma_zx_v(double x, double y, double z, const double* prism);
double gamma_yz_v(double x, double y, double z, const double* prism);
double gamma_zy_v(double x, double y, double z, const double* prism);
#ifdef __cplusplus
}
#endif
	

#endif /* GAMMA_AB_V_H */
