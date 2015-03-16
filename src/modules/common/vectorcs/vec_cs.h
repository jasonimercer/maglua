#include <luabaseobject.h>

#ifndef VECCS
#define VECCS

// support class and functions for coordinate systems

// Cartesian = x,y,z.
// Spherical = r,phi,theta:  physics convention. (radial, azimuthal, polar) r in R, phi in [0:2pi), theta in [0:pi]
// Canonical = r,phi,p:      r in R, phi in [0:2pi), p in [-1:1] = cos(theta)

enum CoordinateSystem
{
	Undefined =-1,
	Cartesian = 0,
	Spherical = 1,
	Canonical = 2,
	SphericalX = 3,
	SphericalY = 4,
	CanonicalX = 5,
	CanonicalY = 6
};

const char* nameOfCoordinateSystem(CoordinateSystem cs);
CoordinateSystem coordinateSystemByName(const char* name);

class VectorCS : public LuaBaseObject
{
public:
    VectorCS() 
        : LuaBaseObject(hash32(VectorCS::slineage(0))) 
    {
        set(0,0,0);
    }
    VectorCS(double v0, double v1, double v2, CoordinateSystem CS=Cartesian)
        : LuaBaseObject(hash32(VectorCS::slineage(0))) 
    {
        set(v0,v1,v2,CS);
    }
    VectorCS(const double* vv, CoordinateSystem CS=Cartesian)
        : LuaBaseObject(hash32(VectorCS::slineage(0))) 
    {
        set(vv[0], vv[1], vv[2], CS);
    }
    VectorCS(const VectorCS& p)
        : LuaBaseObject(hash32(VectorCS::slineage(0))) 
    {
        set(p.v[0], p.v[1], p.v[2], p.cs);
    }

    void set(double v0, double v1, double v2, CoordinateSystem c=Cartesian)
    {
        v[0] = v0; v[1] = v1; v[2] = v2; cs = c; fix();        
    }

    VectorCS& operator= (const VectorCS& p)
    {
        if(this == &p)
            return *this;
        v[0] = p.v[0]; v[1] = p.v[1]; v[2] = p.v[2]; cs = p.cs;
        return *this;
    }
    VectorCS& operator+=(const VectorCS& other)
    {
        if(other.cs != cs)
        {
            return (*this) += other.convertedToCoordinateSystem(cs);
        }

        v[0] += other.v[0];
        v[1] += other.v[1];
        v[2] += other.v[2];
        return (*this);
    }

    VectorCS& operator+(const VectorCS& other)
    {
        VectorCS v(*this);
        return (v += other);
    }

    VectorCS copy() const
    {
        return VectorCS(v,cs);
    }
    

    LINEAGE1("VectorCS");
    static const luaL_Reg* luaMethods();
    virtual int luaInit(lua_State* L, int base=1);
    static int help(lua_State* L);
    virtual void encode(buffer* b);
    virtual int  decode(buffer* b);



    
    VectorCS normalizedTo(double length=1.0) const;
    
    void convertToCoordinateSystem(CoordinateSystem new_cs);
    VectorCS convertedToCoordinateSystem(CoordinateSystem new_cs) const;
    bool isType(CoordinateSystem test) const;

    double magnitude() const;
    VectorCS& setMagnitude(double m);
    
    void scaleFactors(double* sf);
    void stepSize(const double epsilon, double* sf);
    // add a random cartesian vector (scaled by m) to the cartesian 
    // for of the current vector. Rescale to old length, convert back to original CS.
    void randomizeDirection(double m);
    VectorCS randomizedDirection(double m) const;
    
    void scale(double s);
    VectorCS scaled(double s) const;
    
    void scale(double* s3);
    VectorCS scaled(double* s3) const;
    
    void project(const VectorCS& b);
    VectorCS projected(const VectorCS& b) const;
    
    void reject(const VectorCS& b);
    VectorCS rejected(const VectorCS& b) const;
    
    void zeroRadialComponent();
    VectorCS zeroedRadialComponent() const;
    
    void zeroRadialComponent(const VectorCS& radial);
    VectorCS zeroedRadialComponent(const VectorCS& radial) const;
    
    static double angleBetween(const VectorCS& v1, const VectorCS& v2);
    static double arcLength(const VectorCS& v1, const VectorCS& v2);
    static VectorCS cross(const VectorCS& v1, const VectorCS& v2);
    static VectorCS axpy(const double alpha, const VectorCS& x, const VectorCS& y);
    
    void rotateAboutBy(const VectorCS& n, const double t);
    VectorCS rotatedAboutBy(const VectorCS& n, const double t) const;
	
    double v[3];
    CoordinateSystem cs;
    
private:
    void fix(); // fix ranges so they look good 
};


CoordinateSystem baseType(CoordinateSystem cs);

VectorCS lua_toVectorCS(lua_State* L, int base);
VectorCS lua_toVectorCS(lua_State* L, int base, int& consume);
int lua_toVectorCS(lua_State* L, int base, vector<VectorCS>& vv);

int lua_toVVectorCS(lua_State* L, int base, vector<VectorCS>& vv);


void convertCoordinateSystem(CoordinateSystem src_cs, CoordinateSystem dest_cs, const double* src, double* dest);

// VectorCS flags
#define VCSF_ASTABLE 0x1 /* use a lua table */
#define VCSF_CSDESC  0x2 /* string form of the coordinate system name appended */
int lua_pushVectorCS(lua_State* L, const VectorCS& v, const int flags=0x0);
int lua_pushVVectorCS(lua_State* L, std::vector<VectorCS>& vv);

// debug
#include <vector>
void print_vec(const char* msg, const VectorCS& v);
void print_vec(const VectorCS& v);
void print_vecv(std::vector<VectorCS>& vv);

#endif

