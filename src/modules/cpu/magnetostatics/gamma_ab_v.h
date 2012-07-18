#ifndef GAMMA_AB_V_H
#define GAMMA_AB_V_H

#ifdef __cplusplus
extern "C"
{
#endif

double gamma_xx_v(const double x, const double y, const double z, const double* prism);
double gamma_yy_v(const double x, const double y, const double z, const double* prism);
double gamma_zz_v(const double x, const double y, const double z, const double* prism);
double gamma_xy_v(const double x, const double y, const double z, const double* prism);
double gamma_yx_v(const double x, const double y, const double z, const double* prism);
double gamma_xz_v(const double x, const double y, const double z, const double* prism);
double gamma_zx_v(const double x, const double y, const double z, const double* prism);
double gamma_yz_v(const double x, const double y, const double z, const double* prism);
double gamma_zy_v(const double x, const double y, const double z, const double* prism);
#ifdef __cplusplus
}
#endif
	

#endif /* GAMMA_AB_V_H */
