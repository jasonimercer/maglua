#include <math.h>
#include <stdio.h>
#include <string.h>

static inline void G_parts(const double* vecr, double& f3, double& f5)
{
    const double& rx = vecr[0];
    const double& ry = vecr[1];
    const double& rz = vecr[2];

    const double rdotr = rx*rx + ry*ry + rz*rz;

    if(rdotr == 0)
    {
        f3 = 0;
        f5 = 0;
        return;
    }

    const double r = sqrt(rdotr);
    const double r3 = rdotr * r;

    f5 = 1.0 / (r3 * rdotr);
    f3 = f5 * rdotr;
    f5 = - 3.0 * f5;
}


#define make_functionAA(ab, a, b) \
    double __G ## ab(const double* r) { \
        double f3, f5; \
        G_parts(r, f3, f5); \
        return (1.0 / 4.0 * M_PI) * (f3 + r[a]*r[b]*f5); }
#define make_functionAB(ab, a, b) \
    double __G ## ab(const double* r) { \
        double f3, f5; \
        G_parts(r, f3, f5); \
        return (1.0 / 4.0 * M_PI) * (r[a]*r[b]*f5); }

make_functionAA(xx, 0, 0)
make_functionAB(xy, 0, 1)
make_functionAB(xz, 0, 2)

make_functionAB(yx, 1, 0)
make_functionAA(yy, 1, 1)
make_functionAB(yz, 1, 2)

make_functionAB(zx, 2, 0)
make_functionAB(zy, 2, 1)
make_functionAA(zz, 2, 2)



typedef double (*fp)(const double*); // function pointer typedef

// rectangular numerical integration
template <fp Gab>
static double intGab_rect(const double* vsd, const double volume)
{
    const double r[3] = {((vsd[0+6]+vsd[1+6]) - (vsd[0]+vsd[1]))*0.5, 
                         ((vsd[2+6]+vsd[3+6]) - (vsd[2]+vsd[3]))*0.5, 
                         ((vsd[4+6]+vsd[5+6]) - (vsd[4]+vsd[5]))*0.5};

    return Gab(r) * volume;
}

// trapezoidal integration
#define BIT(x, b) (( x & (1<<b)) >> b)
template <fp Gab>
double intGab_trap(const double* vsd, const double volume)
{
    double r[3];
    
    double sum = 0;
    for(unsigned char i=0; i<64; i++)
    {
        r[0] = vsd[ 6+BIT(i,0)] - vsd[0+BIT(i,1)];
        r[1] = vsd[ 8+BIT(i,2)] - vsd[2+BIT(i,3)];
        r[2] = vsd[10+BIT(i,4)] - vsd[4+BIT(i,5)];
        sum += Gab(r);
    }

    return sum * volume / 64.0;
}

template <fp Gab>
double intGab_simp(const double* vsd, const double volume)
{
    const double T = intGab_trap<Gab>(vsd, volume);
    const double M = intGab_rect<Gab>(vsd, volume);

    return (2 * M + T) / 3;
}




static void split6(const double* original, const int split_coord, double* s1, double* s2)
{
    memcpy(s1, original, sizeof(double)*12);
    memcpy(s2, original, sizeof(double)*12);

    const double mid = (original[2*split_coord] + original[2*split_coord+1]) * 0.5;

    s1[2*split_coord+1] = mid;
    s2[2*split_coord+0] = mid;
}


template <fp Gab>
double Nab_(const double* vsd, const double tol, const int depth, 
            const int max_depth, double& coarse_soln, const double volume)
{
    double half1[12];
    double half2[12];

    split6(vsd, depth % 6, half1, half2);

    const double half_volume = volume * 0.5;
    double fine_soln1, fine_soln2;

    if(depth < 2)
    {
        fine_soln1 = intGab_rect<Gab>(half1, half_volume);
        fine_soln2 = intGab_rect<Gab>(half2, half_volume);
    }
    else
    {
        fine_soln1 = intGab_simp<Gab>(half1, half_volume);
        fine_soln2 = intGab_simp<Gab>(half2, half_volume);
    }

    const double fine_soln = fine_soln1 + fine_soln2;

    if(fabs(coarse_soln - fine_soln) < tol)
    {
        // last_value = last_value;
        return fine_soln;
    }

    if((max_depth > -1) && depth >= max_depth)
    {
        return fine_soln;
    }

    const double best_soln1 = Nab_<Gab>(half1, tol, depth+1, max_depth, fine_soln1, half_volume);
    const double best_soln2 = Nab_<Gab>(half2, tol, depth+1, max_depth, fine_soln2, half_volume);

    coarse_soln = fine_soln1 + fine_soln2;

    return best_soln1 + best_soln2;
}

template <fp Gab>
static double Nab(const double* vsd, const double tol, const long int max_depth, double& error)
{
    double v1 = (vsd[1]-vsd[0]) * (vsd[3]-vsd[2]) * (vsd[5]-vsd[4]);
    double v2 = (vsd[7]-vsd[6]) * (vsd[9]-vsd[8]) * (vsd[11]-vsd[10]);

    double volume = 1; // we add the real volume after
    double coarse_soln = intGab_rect<Gab>(vsd, volume);
    const double prefactor = -(v1*v2)/(4.0*M_PI);

    const double better_soln = prefactor * Nab_<Gab>(vsd, tol, 0, max_depth, coarse_soln, 1);

    coarse_soln = prefactor * coarse_soln;

    error = (better_soln - coarse_soln);

    return better_soln;
}



#define make_int_ab(ab) \
    double N##ab(const double* vsd, const double tol, const int max_depth, double& error)\
    {return Nab<__G##ab>(vsd, tol, max_depth, error);}

make_int_ab(xx)
make_int_ab(xy)
make_int_ab(xz)

make_int_ab(yx)
make_int_ab(yy)
make_int_ab(yz)

make_int_ab(zx)
make_int_ab(zy)
make_int_ab(zz)




// From here down we're following Donahue's prescription
// http://math.nist.gov/~MDonahue/talks/hmm2007-MBO-03-accurate_demag.pdf

static double logND(const double num, const double denom)
{
    if(num == 0 || denom == 0)
        return 0;
    return log(num/denom);
}

static double invtan(const double num, const double denom)
{
    if(num == 0 && denom == 0)
        return 0;
    return atan2(num,denom);
}

double ms_f(const double X, const double Y, const double Z)
{
    const double x = fabs(X);
    const double y = fabs(Y);
    const double z = fabs(Z);

    const double xx = x*x;
    const double yy = y*y;
    const double zz = z*z;

    const double R = sqrt(xx+yy+zz);

    const double ypR2 = (y+R)*(y+R);
    const double zpR2 = (z+R)*(z+R);

    return (1.0/12.0) * (
        2 * (2 * xx - yy - zz) * R
        - 12.0 * x * y * z * invtan(y*z, x*R)
        + 3 * y * zz * logND( ypR2, xx+zz )
        - 3 * y * xx * logND( ypR2, xx+zz )
        + 3 * z * yy * logND( zpR2, xx+yy )
        - 3 * z * xx * logND( zpR2, xx+yy )
        );
}


double ms_g(const double X, const double Y, const double Z)
{
    const double x = fabs(X);
    const double y = fabs(Y);
    const double z = fabs(Z);

    const double xx = x*x;
    const double yy = y*y;
    const double zz = z*z;

    const double R = sqrt(xx+yy+zz);

    const double xpR2 = (x+R)*(x+R);
    const double ypR2 = (y+R)*(y+R);
    const double zpR2 = (z+R)*(z+R);

    const double s1 = (X<0)?-1:1;
    const double s2 = (Y<0)?-1:1;

    return (s1*s2/6.0) * (
        - 2 * x * y * R
        - 3 * z * (xx * invtan(y*z, x*R) + yy * invtan(x*z, y*R) + (1.0/3.0) * zz * invtan(x*y, z*R))
        + 3 * x * y * z * logND(zpR2, xx+yy)
        + 0.5 * y * (3 * zz - yy) * logND(xpR2, yy+zz)
        + 0.5 * x * (3 * zz - xx) * logND(ypR2, xx+zz)
        );
}



// Ns is called L in Donahue's talk but L is often the Lua State
typedef double (*dfddd)(const double, const double, const double);
template <dfddd s>
static double Ns(const double X,   const double Y,   const double Z,
                 const double dx1, const double dy1, const double dz1,
                 const double dx2, const double dy2, const double dz2)
{
    const double sign[4]={1.0, -1.0, 1.0, -1.0};
    const double xx[4] = {0, dx2, dx2-dx1, -dx1};
    const double yy[4] = {0, dy2, dy2-dy1, -dy1};
    const double zz[4] = {0, dz2, dz2-dz1, -dz1};
    double sum = 0;

    for(int v1=0; v1<4; v1++)
    {
        for(int v2=0; v2<4; v2++)
        {
            for(int v3=0; v3<4; v3++)
            {
                sum += sign[v1]*sign[v2]*sign[v3]*s(X+xx[v1], Y+yy[v2], Z+zz[v3]);
            }
        }
    }
    return sum / (4.0 * M_PI * dx2 * dy2 * dz2);
}

double magnetostatic_Nxx(const double X,   const double Y,   const double Z,
                         const double dx1, const double dy1, const double dz1,
                         const double dx2, const double dy2, const double dz2)
{
    return Ns<ms_f>(X,Y,Z, dx1,dy1,dz1, dx2,dy2,dz2);
}

double magnetostatic_Nyy(const double X,   const double Y,   const double Z,
                         const double dx1, const double dy1, const double dz1,
                         const double dx2, const double dy2, const double dz2)
{
    return magnetostatic_Nxx(Y,X,Z, dy1,dx1,dz1, dy2,dx2,dz2);
}

double magnetostatic_Nzz(const double X,   const double Y,   const double Z,
                         const double dx1, const double dy1, const double dz1,
                         const double dx2, const double dy2, const double dz2)
{
    return magnetostatic_Nxx(Z,Y,X, dz1,dy1,dx1, dz2,dy2,dx2);
}


double magnetostatic_Nxy(const double X,   const double Y,   const double Z,
                         const double dx1, const double dy1, const double dz1,
                         const double dx2, const double dy2, const double dz2)
{
    return Ns<ms_g>(X,Y,Z, dx1,dy1,dz1, dx2,dy2,dz2);
}

double magnetostatic_Nxz(const double X,   const double Y,   const double Z,
                         const double dx1, const double dy1, const double dz1,
                         const double dx2, const double dy2, const double dz2)
{
    return magnetostatic_Nxy(X,Z,Y, dx1,dz1,dy1, dx2,dz2,dy2);
}

double magnetostatic_Nyx(const double X,   const double Y,   const double Z,
                         const double dx1, const double dy1, const double dz1,
                         const double dx2, const double dy2, const double dz2)
{
    return magnetostatic_Nxy(X,Y,Z, dx1,dy1,dz1, dx2,dy2,dz2);
}

double magnetostatic_Nyz(const double X,   const double Y,   const double Z,
                         const double dx1, const double dy1, const double dz1,
                         const double dx2, const double dy2, const double dz2)
{
    return magnetostatic_Nxy(Y,Z,X, dy1,dz1,dx1, dy2,dz2,dx2);
}

double magnetostatic_Nzx(const double X,   const double Y,   const double Z,
                         const double dx1, const double dy1, const double dz1,
                         const double dx2, const double dy2, const double dz2)
{
    return magnetostatic_Nxz(X,Y,Z, dx1,dy1,dz1, dx2,dy2,dz2);
}

double magnetostatic_Nzy(const double X,   const double Y,   const double Z,
                         const double dx1, const double dy1, const double dz1,
                         const double dx2, const double dy2, const double dz2)
{
    return magnetostatic_Nyz(X,Y,Z, dx1,dy1,dz1, dx2,dy2,dz2);
}


extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}




static int l_NXX1(lua_State* L)
{
    double n[9];
    for(int i=0; i<9; i++)
        n[i] = lua_tonumber(L, i+1);

    lua_pushnumber(L, magnetostatic_Nxx(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8]));
    return 1;
}
static int l_NXY1(lua_State* L)
{
    double n[9];
    for(int i=0; i<9; i++)
        n[i] = lua_tonumber(L, i+1);

    lua_pushnumber(L, magnetostatic_Nxy(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8]));
    return 1;
}
static int l_NXZ1(lua_State* L)
{
    double n[9];
    for(int i=0; i<9; i++)
        n[i] = lua_tonumber(L, i+1);

    lua_pushnumber(L, magnetostatic_Nxz(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8]));
    return 1;
}


static int l_NYX1(lua_State* L)
{
    double n[9];
    for(int i=0; i<9; i++)
        n[i] = lua_tonumber(L, i+1);

    lua_pushnumber(L, magnetostatic_Nyx(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8]));
    return 1;
}
static int l_NYY1(lua_State* L)
{
    double n[9];
    for(int i=0; i<9; i++)
        n[i] = lua_tonumber(L, i+1);

    lua_pushnumber(L, magnetostatic_Nyy(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8]));
    return 1;
}
static int l_NYZ1(lua_State* L)
{
    double n[9];
    for(int i=0; i<9; i++)
        n[i] = lua_tonumber(L, i+1);

    lua_pushnumber(L, magnetostatic_Nyz(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8]));
    return 1;
}


static int l_NZX1(lua_State* L)
{
    double n[9];
    for(int i=0; i<9; i++)
        n[i] = lua_tonumber(L, i+1);

    lua_pushnumber(L, magnetostatic_Nzx(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8]));
    return 1;
}
static int l_NZY1(lua_State* L)
{
    double n[9];
    for(int i=0; i<9; i++)
        n[i] = lua_tonumber(L, i+1);

    lua_pushnumber(L, magnetostatic_Nzy(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8]));
    return 1;
}
static int l_NZZ1(lua_State* L)
{
    double n[9];
    for(int i=0; i<9; i++)
        n[i] = lua_tonumber(L, i+1);

    lua_pushnumber(L, magnetostatic_Nzz(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8]));
    return 1;
}



static void make_vsd(double* n, double* vsd)
{
    vsd[0] = 0; vsd[1] = n[3];
    vsd[2] = 0; vsd[3] = n[4];
    vsd[4] = 0; vsd[5] = n[5];

    vsd[6] = n[0]; vsd[7] = n[0]+n[6];
    vsd[8] = n[1]; vsd[9] = n[1]+n[7];
    vsd[10]= n[2]; vsd[11]= n[2]+n[8];
}

template <fp Gab>
int l_NAB(lua_State* L)
{
    double error;
    double n[9];
    for(int i=0; i<9; i++)
        n[i] = lua_tonumber(L, i+1);

    double tol = lua_tonumber(L, 10);
    double vsd[12];

    make_vsd(n, vsd);

    int max_depth = -1;
    if(lua_isnumber(L, 11))
        max_depth = lua_tointeger(L, 11);


    lua_pushnumber(L, Nab<Gab>(vsd, tol, max_depth, error));
    lua_pushnumber(L, error);
    return 2;

}




void l_mag2d_support_register(lua_State* L)
{
    lua_getglobal(L, "Magnetostatics2D");

    lua_pushcfunction(L, l_NXX1);    lua_setfield(L, -2, "NXX");
    lua_pushcfunction(L, l_NXY1);    lua_setfield(L, -2, "NXY");
    lua_pushcfunction(L, l_NXZ1);    lua_setfield(L, -2, "NXZ");

    lua_pushcfunction(L, l_NYX1);    lua_setfield(L, -2, "NYX");
    lua_pushcfunction(L, l_NYY1);    lua_setfield(L, -2, "NYY");
    lua_pushcfunction(L, l_NYZ1);    lua_setfield(L, -2, "NYZ");

    lua_pushcfunction(L, l_NZX1);    lua_setfield(L, -2, "NZX");
    lua_pushcfunction(L, l_NZY1);    lua_setfield(L, -2, "NZY");
    lua_pushcfunction(L, l_NZZ1);    lua_setfield(L, -2, "NZZ");

    lua_pushcfunction(L, &(l_NAB<__Gxx>));    lua_setfield(L, -2, "NXX_Integrate");
    lua_pushcfunction(L, &(l_NAB<__Gxy>));    lua_setfield(L, -2, "NXY_Integrate");
    lua_pushcfunction(L, &(l_NAB<__Gxz>));    lua_setfield(L, -2, "NXZ_Integrate");

    lua_pushcfunction(L, &(l_NAB<__Gyx>));    lua_setfield(L, -2, "NYX_Integrate");
    lua_pushcfunction(L, &(l_NAB<__Gyy>));    lua_setfield(L, -2, "NYY_Integrate");
    lua_pushcfunction(L, &(l_NAB<__Gyz>));    lua_setfield(L, -2, "NYZ_Integrate");

    lua_pushcfunction(L, &(l_NAB<__Gzx>));    lua_setfield(L, -2, "NZX_Integrate");
    lua_pushcfunction(L, &(l_NAB<__Gzy>));    lua_setfield(L, -2, "NZY_Integrate");
    lua_pushcfunction(L, &(l_NAB<__Gzz>));    lua_setfield(L, -2, "NZZ_Integrate");


    lua_pop(L, 1);
}

