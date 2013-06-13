#include "mep.h"
#include "mep_luafuncs.h"
#include "info.h"
#include "luamigrate.h"
#include <math.h>


// deterministic random number generator - don't want to interfere with
// the internal state of the C generator. Too lazy to use stateful rng
static unsigned long next = 1;
/* RAND_MAX assumed to be 32767 */
int myrand(void) {
	next = next * 1103515245 + 12345;
	return((unsigned)(next/65536) % 32768);
}
double myrandf()
{
	return (double)(myrand()) / 32767.0;
}


static double dot(const double* a, const double* b, int n = 3)
{
	double s = 0;
	for(int i=0; i<n; i++)
		s += a[i]*b[i];
	return s;
}

static void normalizeTo(double* dest, const double* src, const double length=1.0)
{
	double len = dot(src, src);
	if(len == 0)
	{
		dest[0] = length;
		dest[1] = 0;
		dest[2] = 0;
	}
	else
	{
		const double iln = length/sqrt(len);
		dest[0] = iln * src[0];
		dest[1] = iln * src[1];
		dest[2] = iln * src[2];
	}
}


static double angleBetween(const double* a, const double* b)
{
	const double n = sqrt(dot(a,a) * dot(b,b));
	
	if(n == 0)
		return 0;

	const double ct = dot(a,b) / n;
	return acos(ct);
}

static void cross(const double* a, const double* b, double* c)
{
	c[0] = a[1]*b[2] - a[2]*b[1];
	c[1] = a[2]*b[0] - a[0]*b[2];
	c[2] = a[0]*b[1] - a[1]*b[0];
}

// input vector: a
// requires unit length normal vector: n
// input theta: t
// output vector: b
static void rotateAboutBy(const double* a, const double* n, const double t, double* b)
{
	const double ux = n[0];
	const double uy = n[1];
	const double uz = n[2];

	const double sx = a[0];
	const double sy = a[1];
	const double sz = a[2];
	
// 	printf("(%s:%i)\n", __FILE__, __LINE__);
// 	printf("rotate [%g %g %g] about [%g %g %g] by %g\n", a[0], a[1], a[2], n[0], n[1], n[2], t);
	
	// rotating based on Taylor, Camillo; Kriegman (1994). "Minimization on the Lie Group SO(3) and Related Manifolds". Technical Report. No. 9405 (Yale University).
	//
	// 	Here is the rotation matrix:
	const double cost = cos(t);
	const double sint = sin(t);
	
	const double R[3][3] = {
		{ cost + ux*ux * (1.0-cost),   ux*uy*(1.0-cost) - uz*sint, ux*uz * (1.0-cost) + uy * sint},
		{ uy*ux*(1.0-cost) + uz*sint,   cost + uy*uy * (1.0-cost), uy*uz * (1.0-cost) - ux * sint},
		{ uz*ux*(1.0-cost) - uy*sint, uz*uy*(1.0-cost) + ux*sint, cost + uz*uz * (1.0-cost)} };
	
	// now to multiply R * {sx,sy,sz}
	b[0] = R[0][0]*sx + R[1][0]*sy + R[2][0]*sz;
	b[1] = R[0][1]*sx + R[1][1]*sy + R[2][1]*sz;
	b[2] = R[0][2]*sx + R[1][2]*sy + R[2][2]*sz;
	
// 	printf("    = [%g %g %g]\n", b[0], b[1], b[2]);
}


MEP::MEP()
	: LuaBaseObject(hash32(MEP::slineage(0)))
{
	ref_data = LUA_REFNIL;
	energy_ok = false;
}


MEP::~MEP()
{
	deinit();
	
}

void MEP::init()
{
}

void MEP::deinit()
{
	if(ref_data != LUA_REFNIL)
		luaL_unref(L, LUA_REGISTRYINDEX, ref_data);
	ref_data = LUA_REFNIL;

	state_xyz_path.clear();
	sites.clear();
}


int MEP::luaInit(lua_State* L, int base)
{
	state_xyz_path.clear();
	sites.clear();
	
	return LuaBaseObject::luaInit(L, base);
}

void MEP::encode(buffer* b) //encode to data stream
{
	if(ref_data != LUA_REFNIL)
	{
		lua_rawgeti(L, LUA_REGISTRYINDEX, ref_data);
	}
	else
	{
		lua_pushnil(L);
	}

	_exportLuaVariable(L, -1, b);
	lua_pop(L, 1);
	
	encodeInteger(state_xyz_path.size(), b);
	for(int i=0; i<state_xyz_path.size(); i++)
	{
		encodeDouble(state_xyz_path[i], b);
	}
	
	encodeInteger(sites.size(), b);
	for(int i=0; i<sites.size(); i++)
	{
		encodeInteger(sites[i], b);
	}
}

int  MEP::decode(buffer* b) // decode from data stream
{
	deinit();
	_importLuaVariable(L, b);
	ref_data = luaL_ref(L, LUA_REGISTRYINDEX);
	
	int n = decodeInteger(b);
	for(int i=0; i<n; i++)
	{
		state_xyz_path.push_back(decodeDouble(b));
	}
	n = decodeInteger(b);
	for(int i=0; i<n; i++)
	{
		sites.push_back(decodeInteger(b));
	}
	
	return 0;
}


void MEP::randomize(const double magnitude)
{
	double* v = &state_xyz_path[0];

	const int num_sites = numberOfSites();

	// not touching start/end points
	for(int i=num_sites*3; i<state_xyz_path.size() - num_sites*3; i+= 3)
	{
		double x = v[i+0];
		double y = v[i+1];
		double z = v[i+2];
		
		const double m1 = sqrt(x*x + y*y + z*z);
		
		x += (myrandf() * magnitude);
		y += (myrandf() * magnitude);
		z += (myrandf() * magnitude);
		
		const double m2 = sqrt(x*x + y*y + z*z);

		if(m2 > 0)
		{
			const double m = m1/m2;
			v[i+0] = x * m;
			v[i+1] = y * m;
			v[i+2] = z * m;
		}
	}
}



double MEP::distanceBetweenPoints(int p1, int p2, int site)
{
	const int num_sites = sites.size() / 3;

	const double* v1 = &state_xyz_path[p1 * num_sites * 3 + site * 3];
	const double* v2 = &state_xyz_path[p2 * num_sites * 3 + site * 3];
	
	const double d = angleBetween(v1, v2);

	return d;
}

double MEP::distanceBetweenHyperPoints(int p1, int p2)
{
	double d = 0;
	int n = sites.size() / 3;
	for(int i=0; i<n; i++)
	{
		d += distanceBetweenPoints(p1, p2, i);
	}
	return d;
}

void MEP::interpolatePoints(const int p1, const int p2, const int site, const double ratio, vector<double>& dest)
{
	const int num_sites = sites.size() / 3;
	const double* v1 = &state_xyz_path[p1 * num_sites * 3 + site * 3];
	const double* v2 = &state_xyz_path[p2 * num_sites * 3 + site * 3];
	
	double nv1[3];
	double nv2[3];
	
	normalizeTo(nv1, v1);
	normalizeTo(nv2, v2);
	
	
	double a = angleBetween(nv1, nv2);
	
	if(a == 0) //then same vector, no need to interpolate
	{
		dest.push_back(v1[0]);
		dest.push_back(v1[1]);
		dest.push_back(v1[2]);		
		return;
	}
	
	double norm[3];

// 	printf("(%s:%i)  nv1(%i): [%g, %g, %g]\n", __FILE__, __LINE__, p1, nv1[0], nv1[1], nv1[2]);
// 	printf("(%s:%i)  nv2(%i): [%g, %g, %g]\n", __FILE__, __LINE__, p2, nv2[0], nv2[1], nv2[2]);
	
// 	printf(">>> %f\n", fabs(a - 3.1415926538979));
	
	if(fabs(a - 3.1415926538979) < 1e-8) //then colinear, need a random ortho vector
	{
		double t[3]; 
		t[0] = -nv2[2]; // notice the coord twiddle
		t[1] = -nv2[0]; 
		t[2] =  nv2[1];
		cross(nv1,t,norm);

// 		printf("(%s:%i)\n",  __FILE__, __LINE__);
		if(dot(norm, norm) == 0) //then
		{
// 			printf("(%s:%i) RAND!!! \n",  __FILE__, __LINE__);
			t[0] = myrandf()*2.0-1.0;
			t[1] = myrandf()*2.0-1.0;
			t[2] = myrandf()*2.0-1.0;
			cross(nv1,t,norm); //assuming this works
		}
	}
	else
	{
// 		printf("(%s:%i)\n",  __FILE__, __LINE__);
		cross(nv1,nv2,norm);
	}

	normalizeTo(norm, norm);

// 	normalizeTo(norm, norm);
// 	printf("(%s:%i) norm [%e %e %e]\n", __FILE__, __LINE__, norm[0], norm[1], norm[2]);
	
	
	double res[3];

	
	rotateAboutBy(nv1, norm, -a*ratio, res);
	
	normalizeTo(res, res, sqrt(dot(v1,v1)));
	
	dest.push_back(res[0]);
	dest.push_back(res[1]);
	dest.push_back(res[2]);
}



void MEP::interpolateHyperPoints(const int p1, const int p2, const double ratio, vector<double>& dest)
{
	int n = sites.size() / 3;
	for(int i=0; i<n; i++)
	{
		interpolatePoints(p1,p2,i,ratio,dest);
	}
}


// get interval that bounds the distance
static int get_interval(const vector<double>& v, double x, double& ratio)
{
// 	printf("x = %f\n", x);	
	for(unsigned int i=0; i<v.size(); i++)
	{
		const double y = v[i];
		if(x < y)
		{
			ratio = x/y;
			return i;
		}
		x -= y;
	}
	return -1;
}

int MEP::resampleStateXYZPath(lua_State* L, int new_num_points)
{
// 	printf("resample from %i\n", state_xyz_path.size());

	if(new_num_points < 2)
	{
		return luaL_error(L, "attempting to resample path to less than 2 points");
	}
	
	vector<double> new_state_xyz_path;
	const int num_points = numberOfSites();
	double distance = 0; //total distance

	vector<double> d12;
	
	const int num_hyperpoints = numberOfPoints();
	
	for(int i=0; i<num_hyperpoints-1; i++)
	{
		const double d = distanceBetweenHyperPoints(i,i+1);
		d12.push_back(d);
		distance += d;
	}
	
	// endpoints are fixed: path start
	for(int i=0; i<sites.size(); i++)
	{
		new_state_xyz_path.push_back( state_xyz_path[i] );
	}
// 	printf("distance = %f\n", distance);
	const double interval = distance / ((double)(new_num_points-1));
	double ratio;
	
	for(int i=1; i<new_num_points-1; i++)
	{
		int j = get_interval(d12, interval * i, ratio);
		if(j != -1)
		{
// 			printf("range[ %i:%i]  ratio: %e\n", j, j+1, ratio); 
			interpolateHyperPoints(j, j+1, ratio, new_state_xyz_path);
		}
	}
	
	// endpoints are fixed: path end
	const int last_set_idx = state_xyz_path.size() - sites.size();
	for(int i=0; i<sites.size(); i++)
	{
		//printf("last: %f\n", state_xyz_path[last_set_idx + i]);
		new_state_xyz_path.push_back( state_xyz_path[last_set_idx + i] );
	}
	
// 	printf("%i %i\n", state_xyz_path.size(), new_state_xyz_path.size());
	
	state_xyz_path.clear();
	for(unsigned int i=0; i<new_state_xyz_path.size(); i++)
	{
		state_xyz_path.push_back(new_state_xyz_path[i]);
	}
	
	energy_ok = false; //need to recalc energy

// 	printf("resample to %i\n", state_xyz_path.size());
}

void MEP::addSite(int x, int y, int z)
{
	sites.push_back(x);
	sites.push_back(y);
	sites.push_back(z);
	energy_ok = false; //need to recalc energy
}

 //project gradients onto vector perpendicular to spin direction
 void MEP::projForcePerpSpins(lua_State* L, int get_index, int set_index, int energy_index)
{
	double proj[3];
	for(int i=0; i<state_xyz_path.size() / 3; i++)
	{
		double* force = &force_vector[i*3];
		double* v = &state_xyz_path[i*3];
		
		const double bb = dot(v, v);
		if(bb == 0)
		{
			proj[0] = 0;
			proj[1] = 0;
			proj[2] = 0;
		}
		else
		{
			const double ab = dot(force, v);
			proj[0] = v[0] * ab/bb;
			proj[1] = v[1] * ab/bb;
			proj[2] = v[2] * ab/bb;
		}

		force[0] -= proj[0];
		force[1] -= proj[1];
		force[2] -= proj[2];
	}
}

int MEP::applyForces(lua_State* L, double dt)
{
	if(force_vector.size() != state_xyz_path.size())
		return luaL_error(L, "forces are not computed or size mismatch");
	
	double absMovement = 0;
	
	double max_m = 0;
	double max_f = 0;
	
	const int num_sites = numberOfSites();
	
	for(int i=0; i<(force_vector.size()/3); i++)
	{
		const double* xyz = &(state_xyz_path[i*3+0]);
		const double m = sqrt(dot(xyz, xyz));

		if(m > max_m)
			max_m = m;
		
		const double* fxyz = &(force_vector[i*3+0]);
		const double f = sqrt(dot(fxyz, fxyz));

		if(f > max_f)
			max_f = f;
	}
	
	// interpretting dt as max motion
	double f_scale = 1.0;
	
	if(max_f > (max_m * dt))
		f_scale = (max_m * dt) / max_f;
	
	// end points are fixed
	for(int i=num_sites; i<(force_vector.size()/3)-num_sites; i++)
	{
		double x = state_xyz_path[i*3+0];
		double y = state_xyz_path[i*3+1];
		double z = state_xyz_path[i*3+2];
		
		const double m = sqrt(x*x+y*y+z*z);
		
		const double dx = force_vector[i*3+0] * f_scale;
		const double dy = force_vector[i*3+1] * f_scale;
		const double dz = force_vector[i*3+2] * f_scale;
		
		x += dx;
		y += dy;
		z += dz;
		
		const double dd = sqrt(dx*dx+dy*dy+dz*dz);
// 		printf("dd: %g    dt m: %g\n", dd, dt*m);
		absMovement += dd;

		const double m2 = sqrt(x*x+y*y+z*z);

		if(m2 > 0)
		{
			x *= m/m2;
			y *= m/m2;
			z *= m/m2;
		}
		
		state_xyz_path[i*3+0] = x;
		state_xyz_path[i*3+1] = y;
		state_xyz_path[i*3+2] = z;
	}
	
	lua_pushnumber(L, sqrt(absMovement));
	
	return 1;
}
	


void MEP::calculateOffsetVector(double* vec, const int p1, const int p2)
{
	const int ss = sites.size();
	for(int i=0; i<ss; i++)
	{
		vec[i] = state_xyz_path[p1*ss+i] - state_xyz_path[p2*ss+i];
	}
}

void MEP::computeTangent(const int p1, const int p2, const int dest)
{
	const int s = sites.size();
	
	// compute difference
 	for(int i=0; i<s; i++)
	{
		path_tangent[i + dest*s] = state_xyz_path[i + p1*s] - state_xyz_path[i + p2*s];
	}

	const int n = state_xyz_path.size();
	for(int i=0; i<n/3; i++)
	{
		const double x = state_xyz_path[i*3 + 0];
		const double y = state_xyz_path[i*3 + 1];
		const double z = state_xyz_path[i*3 + 2];

		double length = sqrt(x*x + y*y + z*z);

		if(length != 0)
		{
			length = 1.0/length;
			path_tangent[i*3 + 0] *= length;
			path_tangent[i*3 + 1] *= length;
			path_tangent[i*3 + 2] *= length;
		}
	}
}


/*
void MEP::computeTangent(lua_State* L, int get_index, int set_index, int energy_index, const int _p)
{
	int p[3];
	p[0] = _p-1;
	p[1] = _p;
	p[2] = _p+1;
	
	if(p[0] < 0)
		p[0] = 0;
	if(p[2] > numberOfPoints())
		p[2] = numberOfPoints();
	
	calculateEnergies(L, get_index, set_index, energy_index);
// 	
	double e[3];
	for(int i=0; i<3; i++)
		e[i] = energies[p[i]];
// 	
	int p1 = p[0];
	int p2 = p[2];
// 	
	if( (e[0] < e[1] && e[2] < e[1]) || (e[0] > e[1] && e[2] > e[1])) // if middle point is ^ or v
	{
		p1 = p[0];
		p2 = p[2];
	}
	else
	{
		if(e[0] > e[1])
		{
			p1 = 0;
			p2 = 1;
		}
		else
		{
			p1 = 1;
			p2 = 2;
		}
	}
	
	const int s = sites.size();
	const int dest = _p;
	
	double tan_len2 = 0;
	// compute difference
 	for(int i=0; i<s; i++)
	{
		const double t = state_xyz_path[i + p1*s] - state_xyz_path[i + p2*s];
		path_tangent[i + dest*s] = t;
		tan_len2 += t*t;
	}
	
	if(tan_len2 > 0)
	{
		tan_len2 = 1.0 / sqrt(tan_len2);
	}
	
	if(energies[p1] < energies[p2])
		tan_len2 *= -1.0;
	
 	for(int i=0; i<s; i++)
	{
		path_tangent[i + dest*s] *= tan_len2;
	}
	

// 	const int n = state_xyz_path.size();
// 	for(int i=0; i<n/3; i++)
// 	{
// 		const double x = state_xyz_path[i*3 + 0];
// 		const double y = state_xyz_path[i*3 + 1];
// 		const double z = state_xyz_path[i*3 + 2];
// 
// 		double length = sqrt(x*x + y*y + z*z);
// 
// 		if(length != 0)
// 		{
// 			length = 1.0/length;
// 			path_tangent[i*3 + 0] *= length;
// 			path_tangent[i*3 + 1] *= length;
// 			path_tangent[i*3 + 2] *= length;
// 		}
// 	}
}*/


void MEP::projForcePerpPath(lua_State* L, int get_index, int set_index, int energy_index) //project gradients onto vector perpendicular to path direction
{
	path_tangent.clear();
	path_tangent.resize(state_xyz_path.size());

	const int s = numberOfSites();

	computeTangent(0, 1, 0);
	for(int i=1; i<s-1; i++)
	{
		computeTangent(i-1,i+1,i);
	}
	computeTangent(s-2, s-1, s-1);



	const int ss = sites.size(); // point size
	const int nn = numberOfPoints(); // number of points

	double* proj = new double[ss];

	for(int i=0; i<nn; i++)
	{
		double* force = &force_vector[i*ss];
		double* v = &path_tangent[i*ss];
		
		const double bb = dot(v, v, ss);
		if(bb == 0)
		{
			for(int i=0; i<ss; i++)
				proj[i] = 0;
		}
		else
		{
			const double ab = dot(force, v, ss);
			for(int i=0; i<ss; i++)
				proj[i] = v[i] * ab/bb;;
		}
		
		for(int i=0; i<ss; i++)
			force[i] -= proj[i];
	}

	delete [] proj;
}



void MEP::projForcePath(lua_State* L, int get_index, int set_index, int energy_index) //project gradients onto vector perpendicular to path direction
{
	path_tangent.clear();
	path_tangent.resize(state_xyz_path.size());

	const int s = numberOfSites();

	computeTangent(0, 1, 0);
	for(int i=1; i<s-1; i++)
	{
		computeTangent(i-1,i+1,i);
	}
	computeTangent(s-2, s-1, s-1);



	const int ss = sites.size(); // point size
	const int nn = numberOfPoints(); // number of points

	double* proj = new double[ss];

	for(int i=0; i<nn; i++)
	{
		double* force = &force_vector[i*ss];
		double* v = &path_tangent[i*ss];
		
		const double bb = dot(v, v, ss);
		if(bb == 0)
		{
			for(int i=0; i<ss; i++)
				proj[i] = 0;
		}
		else
		{
			const double ab = dot(force, v, ss);
			for(int i=0; i<ss; i++)
				proj[i] = v[i] * ab/bb;;
		}
		
		for(int i=0; i<ss; i++)
			force[i] = proj[i];
	}

	delete [] proj;
}

// back up old sites so we can restore after
void MEP::saveConfiguration(lua_State* L, int get_index, vector<double>& buffer)
{
	buffer.clear();
	const int num_sites = numberOfSites();
	
	for(int s=0; s<num_sites; s++)
	{
		getSiteSpin(L, get_index, &sites[s*3+0], buffer);
	}
}

// restore state
void MEP::loadConfiguration(lua_State* L, int set_index, vector<double>& buffer)
{
	const int num_sites = numberOfSites();

	for(int s=0; s<num_sites; s++)
	{
		setSiteSpin(L, set_index, &sites[s*3], &buffer[s*3]);
	}
	
	energy_ok = false; //need to recalc energy
}

	
void MEP::getSiteSpin(lua_State* L, int get_index, int* site3, double* m3)
{
	lua_pushvalue(L, get_index);
	lua_pushinteger(L, sites[0]+1);
	lua_pushinteger(L, sites[1]+1);
	lua_pushinteger(L, sites[2]+1);
	lua_call(L, 3, 3);

	m3[0] = lua_tonumber(L, -3);
	m3[1] = lua_tonumber(L, -2);
	m3[2] = lua_tonumber(L, -1);
	
	lua_pop(L, 3);	
}

void MEP::getSiteSpin(lua_State* L, int get_index, int* site3, vector<double>& v)
{
	double m3[3];
	getSiteSpin(L, get_index, site3, m3);
	v.push_back(m3[0]);
	v.push_back(m3[1]);
	v.push_back(m3[2]);
}


void MEP::setSiteSpin(lua_State* L, int set_index, int* site3, double* m3)
{
	lua_pushvalue(L, set_index);
	lua_pushinteger(L, site3[0]+1);
	lua_pushinteger(L, site3[1]+1);
	lua_pushinteger(L, site3[2]+1);
	lua_pushnumber(L, m3[0]);
	lua_pushnumber(L, m3[1]);
	lua_pushnumber(L, m3[2]);
	lua_call(L, 6, 0);	
	energy_ok = false; //need to recalc energy
}

void MEP::setAllSpins(lua_State* L, int set_index, double* m)
{
	int num_sites = numberOfSites();
	for(int s=0; s<num_sites; s++)
		setSiteSpin(L, set_index, &sites[s*3], &m[s*3]);
}

void MEP::getAllSpins(lua_State* L, int get_index, double* m)
{
	int num_sites = numberOfSites();
	for(int s=0; s<num_sites; s++)
		getSiteSpin(L, get_index, &sites[s*3], &m[s*3]);
}

void MEP::setSiteSpin(lua_State* L, int set_index, int* site3, double sx, double sy, double sz)
{
	double m3[3];
	m3[0] = sx;
	m3[1] = sy;
	m3[2] = sz;
	
	setSiteSpin(L, set_index, site3, m3);
}


double MEP::getEnergy(lua_State* L, int energy_index)
{
	lua_pushvalue(L, energy_index);
	lua_call(L, 0, 1);
	double e = lua_tonumber(L, -1);
	lua_pop(L, 1);
	return e;
}


int MEP::calculateEnergies(lua_State* L, int get_index, int set_index, int energy_index)
{
	if(energy_ok)
		return 0;
		
	energies.clear();
	const int num_sites = numberOfSites();
	const int path_length = numberOfPoints();

	// back up old sites so we can restore after
	vector<double> cfg;
	saveConfiguration(L, get_index, cfg);


	for(int p=0; p<path_length; p++)
	{
		double e = 0;
		for(int s=0; s<num_sites; s++)
		{
			setSiteSpin(L, set_index, &sites[s*3], &state_xyz_path[p * sites.size() + s*3]);
		}

		energies.push_back(getEnergy(L, energy_index));
	}

	loadConfiguration(L, set_index, cfg);
	
	energy_ok = true;
	return 0;
}


static void arrayCopyWithElementChange(double* dest, double* src, int element, double delta, int n)
{
	for(int i=0; i<n; i++)
	{
		dest[i] = src[i];
		if(i == element)
		{
			dest[i] += delta;
		}
	}
}

double MEP::problemScale()
{
	double n = 0;
	double d = 0;
	for(int i=0; i<state_xyz_path.size()/3; i++)
	{
		const double* x = & state_xyz_path[i*3];
		const double m = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
		d += sqrt(m);
		n++;
	}
	
	if(n == 0)
		return 1.0;
	return d/n;
}


double MEP::computePointSecondDerivativeAB(lua_State* L, int p, double __h, int set_index, int get_index, int energy_index, int c1, int c2)
{
	// back up old sites so we can restore after
	vector<double> cfg;
	saveConfiguration(L, get_index, cfg);
	const int num_sites = numberOfSites();
	double* vxyz = &state_xyz_path[p*num_sites*3];

	const double m1 = sqrt(vxyz[0]*vxyz[0] + vxyz[1]*vxyz[1] + vxyz[2]*vxyz[2]);
	const double h = 1e-6*m1;
	
	
	double* state = new double[num_sites * 3];
	
	// calc upper deriv energies
	arrayCopyWithElementChange(state,  vxyz, c1, h, num_sites * 3);
	arrayCopyWithElementChange(state, state, c2, h, num_sites * 3);
	setAllSpins(L, set_index, state);
	const double e1 = getEnergy(L, energy_index);

	arrayCopyWithElementChange(state,  vxyz, c1, h, num_sites * 3);
	arrayCopyWithElementChange(state, state, c2,-h, num_sites * 3);
	setAllSpins(L, set_index, state);
	const double e2 = getEnergy(L, energy_index);


	// calc lower deriv energies
	arrayCopyWithElementChange(state,  vxyz, c1,-h, num_sites * 3);
	arrayCopyWithElementChange(state, state, c2, h, num_sites * 3);
	setAllSpins(L, set_index, state);
	const double e3 = getEnergy(L, energy_index);

	arrayCopyWithElementChange(state,  vxyz, c1,-h, num_sites * 3);
	arrayCopyWithElementChange(state, state, c2,-h, num_sites * 3);
	setAllSpins(L, set_index, state);
	const double e4 = getEnergy(L, energy_index);

	
	const double result = ((e1-e2) - (e3-e4)) / (4.0 * h*h);

	delete [] state;
	
	loadConfiguration(L, set_index, cfg);	
	
	return result;
}


// partial derivs
void MEP::computePointSecondDerivative(lua_State* L, int p, double h, int set_index, int get_index, int energy_index, double* derivsAB)
{
	const int num_sites = numberOfSites();
	int deriv_pos = 0;
	for(int c1=0; c1<num_sites*3; c1++)
	{
		for(int c2=0; c2<num_sites*3; c2++)
		{
			derivsAB[deriv_pos] = computePointSecondDerivativeAB(L, p, h, set_index, get_index, energy_index, c1, c2);
			deriv_pos++;
		}
	}
}

	
// relax individual point. This is used to refine a maximal point
// expected on the stack:
// at 1, mep
// at 2, point number
// at 3, function get_site_ss1(x,y,z) return sx,sy,sz 
// at 4, function set_site_ss1(x,y,z, sx,sy,sz) 
// at 5, function get_energy_ss1()
// at 6, step size (positive to relax down, negative to relax up)
// at 7, number of steps
// at 8, optional direction to negate as a table
// returns energy and energy change due to relax
int MEP::relaxSinglePoint(lua_State* L)
{
	const int get_index = 3;
	const int set_index = 4;
	const int energy_index = 5;

	const int num_sites = numberOfSites();
	const int path_length = numberOfPoints();

	int p = lua_tointeger(L, 2) - 1;
	if(p < 0 || p >= path_length)
		return luaL_error(L, "Point index out of bounds");
	
	if(!lua_isfunction(L, 3) || !lua_isfunction(L, 4) || !lua_isfunction(L, 5))
		return luaL_error(L, "3 functions expected - perhaps you should call the wrapper method :relaxSinglePoint");
	
	double dh = lua_tonumber(L, 6);
	
	int num_steps = lua_tointeger(L, 7);
	if(num_steps <= 0)
		return luaL_error(L, "Non-positive number of steps does not make sense.");
	
	// need to save current configuration
	vector<double> cfg;
	saveConfiguration(L, get_index, cfg);

	double* vxyz = &state_xyz_path[p*num_sites*3];

	// write path point as it currently stands
	setAllSpins(L, set_index, vxyz);
		
	
	double initial_energy = getEnergy(L, energy_index);
	
	double* neg_dir = 0;
	if(lua_istable(L, 8))
	{
		neg_dir = new double[3 * num_sites];
		for(int i=0; i<3 * num_sites; i++)
		{
			lua_pushinteger(L, i+1);
			lua_gettable(L, 8);
			neg_dir[i] = lua_tonumber(L, -1);
			lua_pop(L, 1);
		}
	}
	
	double* grad = new double[3 * num_sites];
	// here's the loop over num_steps
	for(int i=0; i<num_steps; i++)
	{
		// first compute e. grad around point
		
		for(int s=0; s<num_sites; s++)
		{
			computePointGradAtSite(L, p, s, set_index, energy_index, &grad[s*3]);
			
// 			printf("g=%e,%e,%e\n", grad[s*3], grad[s*3+2], grad[s*3+2]);

		}

		if(neg_dir)
		{
			const double n2 = dot(neg_dir, neg_dir, 3*num_sites);
			
			const double proj = dot(neg_dir, grad, 3*num_sites);
			
			if(n2 != 0)
			{
				for(int j=0; j<3 * num_sites; j++)
				{
					neg_dir[j] *= proj/n2;
					
					grad[j] -= 2*neg_dir[j];
				}
			}

		}
		
		for(int s=0; s<num_sites*3; s+=3)
		{
			const double m1 = sqrt(vxyz[s+0]*vxyz[s+0] + vxyz[s+1]*vxyz[s+1] + vxyz[s+2]*vxyz[s+2]);
			
			vxyz[s+0] += grad[s+0] * dh;
			vxyz[s+1] += grad[s+1] * dh;
			vxyz[s+2] += grad[s+2] * dh;
			
			const double m2 = sqrt(vxyz[s+0]*vxyz[s+0] + vxyz[s+1]*vxyz[s+1] + vxyz[s+2]*vxyz[s+2]);

			if(m2 > 0)
			{
				const double m = m1/m2;
				vxyz[s+0] *= m;
				vxyz[s+1] *= m;
				vxyz[s+2] *= m;
			}
		}
	}
	delete [] grad;
	
	if(neg_dir)
		delete [] neg_dir;
	
	// write updated cfg
 	setAllSpins(L, set_index, vxyz);
	
	double final_energy = getEnergy(L, energy_index);

	if(energies.size() < path_length)
	{
		energies.clear();
		for(int p=0; p<path_length; p++)
		{
			double e = 0;
			for(int s=0; s<num_sites; s++)
			{
				setSiteSpin(L, set_index, &sites[s*3], &state_xyz_path[p * sites.size() + s*3]);
			}

			energies.push_back(getEnergy(L, energy_index));
		}	
	}
	else
	{
		energies[p] = final_energy;
	}
	
	// need to restore saved configuration to SpinSystem
	loadConfiguration(L, set_index, cfg);

	lua_pushnumber(L, initial_energy);
	lua_pushnumber(L, final_energy);
	lua_pushnumber(L, fabs(initial_energy - final_energy));
	
	energy_ok = false;
	return 3;
}

int MEP::numberOfPoints()
{
	return state_xyz_path.size() / sites.size();
}

int MEP::numberOfSites()
{
	return sites.size() / 3;
}

void MEP::computePointGradAtSite(lua_State* L, int p, int s, int set_index, int energy_index, double* grad3)
{
	double* vxyz = &state_xyz_path[p*numberOfSites()*3 + s*3];
	
	const double m1 = sqrt(vxyz[0]*vxyz[0] + vxyz[1]*vxyz[1] + vxyz[2]*vxyz[2]);
	
	const double epsilon = 1e-6*m1;
	
	// We now have a point vxyz at site p,s
	// time to compute the spatial energy gradient.
	double g[6] = {0,0,0,0,0,0};

	// 7th below is to restore old state
	const double cc[7][3] = {{1,0,0},{-1,0,0},   {0,1,0},{0,-1,0},   {0,0,1},{0,0,-1},   {0,0,0}};
	for(int c=0; c<7; c++)
	{
		double new_mx = vxyz[0] + cc[c][0] * epsilon;
		double new_my = vxyz[1] + cc[c][1] * epsilon;
		double new_mz = vxyz[2] + cc[c][2] * epsilon;
		
		double m2 = sqrt(new_mx*new_mx + new_my*new_my + new_mz*new_mz);

		if(m2 > 0)
		{
			const double m = m1/m2;
			new_mx *= m;
			new_my *= m;
			new_mz *= m;
		}
		
		setSiteSpin(L, set_index, &sites[s*3], new_mx, new_my, new_mz);
		
		if(c < 6) // last iteration resets cfg. No need to calc energy in that case.
		{
			g[c] = getEnergy(L, energy_index);
		}
	}
		
	grad3[0] = (g[1] - g[0]) / (2.0 * epsilon);
	grad3[1] = (g[3] - g[2]) / (2.0 * epsilon);
	grad3[2] = (g[5] - g[4]) / (2.0 * epsilon);
}

void MEP::writePathPoint(lua_State* L, int set_index, double* vxyz)
{
	const int num_sites = numberOfSites();
	
	// write path point as it is given
	for(int s=0; s<num_sites; s++)
	{
		setSiteSpin(L, set_index, &sites[s*3], &vxyz[s*3]);
	}
	
	energy_ok = false;
}

static int addToTable(lua_State* L, int tab_pos, int index, int value)
{
	index++;
	lua_pushinteger(L, index);
	lua_pushinteger(L, value);
	lua_settable(L, tab_pos);
	return index;
}

int MEP::maxpoints(lua_State* L)
{
	const int path_length = numberOfPoints();
	
	lua_newtable(L); //mins
	const int min_idx = lua_gettop(L);

	lua_newtable(L); //maxs
	const int max_idx = lua_gettop(L);
	
	lua_newtable(L); //all
	const int all_idx = lua_gettop(L);
	
	
	if(energies.size() != path_length)
		return luaL_error(L, "Energies not found. You may need to run a single round of :compute");

	int szMin = 0;
	int szMax = 0;
	int szAll = 0;
	
	if(energies[1] > energies[0]) // start at min
	{
		szMin = addToTable(L, min_idx, szMin, 1);
		szAll = addToTable(L, all_idx, szAll, 1);
	}
	else
	{
		szMax = addToTable(L, max_idx, szMax, 1);
		szAll = addToTable(L, all_idx, szAll, 1);
	}
	
	for(int i=1; i<path_length-1; i++)
	{
		const double a = energies[i-1];
		const double b = energies[i];
		const double c = energies[i+1];
		
		if(b<a && b<c) //local minimum
		{
			szMin = addToTable(L, min_idx, szMin, i+1);
			szAll = addToTable(L, all_idx, szAll, i+1);
		}

		if(b>a && b>c) //local maximum
		{
			szMax = addToTable(L, max_idx, szMax, i+1);
			szAll = addToTable(L, all_idx, szAll, i+1);
		}
	}

	if(energies[path_length-2] > energies[path_length-1]) // end at min
		szMin = addToTable(L, min_idx, szMin, path_length);
	else
		szMax = addToTable(L, max_idx, szMax, path_length);

	szAll = addToTable(L, all_idx, szAll, path_length);

	return 3;
}



int MEP::calculateEnergyGradients(lua_State* L, int get_index, int set_index, int energy_index)
{
	const int num_sites = numberOfSites();
	const int path_length = numberOfPoints();
	
	force_vector.resize( state_xyz_path.size() ); //since we're doing random-ish access
	
	// need to save current configuration
	vector<double> cfg;
	saveConfiguration(L, get_index, cfg);
	
	// lets march along the path
	for(int p=0; p<path_length; p++)
	{
		double* vxyz = &state_xyz_path[p*num_sites*3];
		
		// write path point as it currently stands
		writePathPoint(L, set_index, &state_xyz_path[p*num_sites*3]);

		for(int s=0; s<num_sites; s++)
		{
			double grad3[3];
			computePointGradAtSite(L, p, s, set_index, energy_index, grad3);

			force_vector[p*num_sites*3 + s*3 + 0] = grad3[0];
			force_vector[p*num_sites*3 + s*3 + 1] = grad3[1];
			force_vector[p*num_sites*3 + s*3 + 2] = grad3[2];
			
// 			printf("F = %e,%e,%e\n", grad3[0], grad3[1], grad3[2]);
		}
	}
	
	// need to restore saved configuration
	loadConfiguration(L, set_index, cfg);
	
	return 0;
}


static int l_applyforces(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);

	double dt = 0.1;
	if(lua_isnumber(L, 2))
	{
		dt = lua_tonumber(L, 2);
	}
	
	return mep->applyForces(L, dt);	
}







static int l_setdata(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	luaL_unref(L, LUA_REGISTRYINDEX, mep->ref_data);
	lua_pushvalue(L, 2);
	mep->ref_data = luaL_ref(L, LUA_REGISTRYINDEX);
	return 0;
}

static int l_getdata(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	lua_rawgeti(L, LUA_REGISTRYINDEX, mep->ref_data);
	return 1;
}


static int l_addsite(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	
	int site[3] = {0,0,0};
	
	for(int i=0; i<3; i++)
	{
		if(lua_isnumber(L, 2+i))
		{
			site[i] = lua_tointeger(L, 2+i) - 1;
		}
	}
	
	mep->addSite(site[0], site[1], site[2]);
}

static int l_clearsites(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	mep->sites.clear();
	return 0;
}
static int l_clearpath(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	mep->state_xyz_path.clear();
	return 0;
}

static int l_getallsites(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);

	lua_newtable(L);
	
	for(unsigned int i=0; i<mep->sites.size()/3; i++)
	{
		lua_pushinteger(L, i+1);
		lua_newtable(L);
		for(int j=0; j<3; j++)
		{
			lua_pushinteger(L, j+1);
			lua_pushinteger(L, mep->sites[i*3+j]+1);
			lua_settable(L, -3);
		}
		lua_settable(L, -3);
	} 	
	return 1;
}


static int l_addstatexyz(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	
	double x = lua_tonumber(L, 2);
	double y = lua_tonumber(L, 3);
	double z = lua_tonumber(L, 4);
	
	mep->state_xyz_path.push_back(x);
	mep->state_xyz_path.push_back(y);
	mep->state_xyz_path.push_back(z);
	return 0;
}


static int l_resampleStateXYZPath(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);

	const int n = lua_tointeger(L, 2);
	double noise = 0;
	
	return mep->resampleStateXYZPath(L, n);
}

static int l_projForcePerpSpins(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	
	int get_index = 2;
	int set_index = 3;
	int energy_index = 4;
	
	if(!lua_isfunction(L, get_index))
		return luaL_error(L, "1st argument expected to be get_site_ss1(x,y,z) function");
	if(!lua_isfunction(L, set_index))
		return luaL_error(L, "2nd argument expected to be set_site_ss1(x,y,z,sx,sy,sz) function");
	if(!lua_isfunction(L, energy_index))
		return luaL_error(L, "3rd argument expected to be get_energy_ss1() function");
	
	mep->projForcePerpSpins(L, get_index, set_index, energy_index);
	return 0;
}
static int l_projForcePerpPath(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	
	int get_index = 2;
	int set_index = 3;
	int energy_index = 4;
	
	if(!lua_isfunction(L, get_index))
		return luaL_error(L, "1st argument expected to be get_site_ss1(x,y,z) function");
	if(!lua_isfunction(L, set_index))
		return luaL_error(L, "2nd argument expected to be set_site_ss1(x,y,z,sx,sy,sz) function");
	if(!lua_isfunction(L, energy_index))
		return luaL_error(L, "3rd argument expected to be get_energy_ss1() function");
	mep->projForcePerpPath(L, get_index, set_index, energy_index);
	return 0;
}
static int l_projForcePath(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	
	int get_index = 2;
	int set_index = 3;
	int energy_index = 4;
	
	if(!lua_isfunction(L, get_index))
		return luaL_error(L, "1st argument expected to be get_site_ss1(x,y,z) function");
	if(!lua_isfunction(L, set_index))
		return luaL_error(L, "2nd argument expected to be set_site_ss1(x,y,z,sx,sy,sz) function");
	if(!lua_isfunction(L, energy_index))
		return luaL_error(L, "3rd argument expected to be get_energy_ss1() function");
	
	mep->projForcePath(L, get_index, set_index, energy_index);
	return 0;
}

// at 1, mep
// at 2, function get_site(x,y,z) return sx,sy,sz end
// at 3, function set_site(x,y,z, sx,sy,sz) end
// at 4, get energy function
static int l_calculateEnergyGradients(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	
	int get_index = 2;
	int set_index = 3;
	int energy_index = 4;
	
	if(!lua_isfunction(L, get_index))
		return luaL_error(L, "1st argument expected to be get_site_ss1(x,y,z) function");
	if(!lua_isfunction(L, set_index))
		return luaL_error(L, "2nd argument expected to be set_site_ss1(x,y,z,sx,sy,sz) function");
	if(!lua_isfunction(L, energy_index))
		return luaL_error(L, "3rd argument expected to be get_energy_ss1() function");
	
	return mep->calculateEnergyGradients(L, get_index, set_index, energy_index);
}

// at 1, mep
// at 2, function get_site(x,y,z) return sx,sy,sz end
// at 3, function set_site(x,y,z, sx,sy,sz) end
// at 4, get energy function
static int l_calculateEnergies(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	
	int get_index = 2;
	int set_index = 3;
	int energy_index = 4;
	
	if(!lua_isfunction(L, get_index))
		return luaL_error(L, "1st argument expected to be get_site_ss1(x,y,z) function");
	if(!lua_isfunction(L, set_index))
		return luaL_error(L, "2nd argument expected to be set_site_ss1(x,y,z,sx,sy,sz) function");
	if(!lua_isfunction(L, energy_index))
		return luaL_error(L, "3rd argument expected to be get_energy_ss1() function");
	
	return mep->calculateEnergies(L, get_index, set_index, energy_index);
}

static int l_problemscale(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	lua_pushnumber(L, mep->problemScale());
	return 1;
}

static int l_getpathenergy(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);

	lua_newtable(L);
	for(int i=0; i<mep->energies.size(); i++)
	{
		lua_pushinteger(L, i+1);
		lua_pushnumber(L, mep->energies[i]);
		lua_settable(L, -3);
	}
	return 1;
}

static int l_getgradient(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);

	const int p = lua_tointeger(L, 2) - 1;
	const int s = lua_tointeger(L, 3) - 1;

	if(p < 0 || s < 0)
		return 0;

	int idx = mep->sites.size() * p + s*3;

	if(mep->force_vector.size() < idx+2)
		return 0;

	lua_pushnumber(L, mep->force_vector[idx+0]);
	lua_pushnumber(L, mep->force_vector[idx+1]);
	lua_pushnumber(L, mep->force_vector[idx+2]);
	return 3;
}

// expected on the stack at function call
// at 1, mep
// at 2, point number
// at 3, function get_site_ss1(x,y,z) return sx,sy,sz 
// at 4, function set_site_ss1(x,y,z, sx,sy,sz) 
// at 5, function get_energy_ss1()
// at 6, step size (positive to relax down, negative to relax up)
// at 7, number of steps
// this is the internal version
static int l_relaxSinglePoint_(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);

	return mep->relaxSinglePoint(L);
}

// expected on the stack at function call
// at 1, mep
// at 2, point number
// at 3, function get_site_ss1(x,y,z) return sx,sy,sz 
// at 4, function set_site_ss1(x,y,z, sx,sy,sz) 
// at 5, function get_energy_ss1()
// at 6, step size (positive to relax down, negative to relax up)
// at 7, number of steps
// this is the internal version
// static int l_relaxSaddlePoint_(lua_State* L)
// {
// 	LUA_PREAMBLE(MEP, mep, 1);
// 
// 	return mep->relaxSaddlePoint(L);
// }


// expected on the stack at function call
// at 1, mep
// at 2, point number
// at 3, function get_site_ss1(x,y,z) return sx,sy,sz 
// at 4, function set_site_ss1(x,y,z, sx,sy,sz) 
// at 5, function get_energy_ss1()
// this is the internal version
static int l_computepoint2deriv(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);

	const int get_index = 4;
	const int set_index = 5;
	const int energy_index = 6;
	
	int num_sites = mep->numberOfSites();
	
	int p = lua_tointeger(L, 2) - 1;
	
	if(p < 0 || p >= mep->numberOfPoints())
		return luaL_error(L, "Invalid point number");

	double* derivsAB = new double[num_sites * num_sites * 9];
	
	double h = lua_tonumber(L, 3);

	mep->computePointSecondDerivative(L, p, h, set_index, get_index, energy_index, derivsAB);
	
	lua_newtable(L);
	for(int i=0; i<num_sites * num_sites * 9; i++)
	{
		lua_pushinteger(L, i+1);
		lua_pushnumber(L, derivsAB[i]);
		lua_settable(L, -3);
	}
	return 1;
}	
	
	
static int l_getsite(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);

	const int p = lua_tointeger(L, 2) - 1;
	const int s = lua_tointeger(L, 3) - 1;

	if(p < 0 || s < 0)
	{
		return luaL_error(L, "Path Point or site is out of bounds. {Point,Site} = {%d,%d}. Upper Bound = {%d,%d}", p+1,s+1,mep->numberOfPoints(), mep->numberOfSites());
	}

	int idx = mep->sites.size() * p + s*3;

	if(mep->state_xyz_path.size() < idx+2)
	{
		return luaL_error(L, "Path Point or site is out of bounds. {Point,Site} = {%d,%d}. Upper Bound = {%d,%d}", p+1,s+1,mep->numberOfPoints(), mep->numberOfSites());
	}
	
	const double x = mep->state_xyz_path[idx+0];
	const double y = mep->state_xyz_path[idx+1];
	const double z = mep->state_xyz_path[idx+2];
	const double m = sqrt(x*x + y*y + z*z);
	
	lua_pushnumber(L, x);
	lua_pushnumber(L, y);
	lua_pushnumber(L, z);
	lua_pushnumber(L, m);
	return 4;
}


static int l_sitecount(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);

	lua_pushinteger(L, mep->numberOfSites());
	
	return 1;
}
static int l_randomize_(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	mep->randomize(lua_tonumber(L, 2));
	return 0;
}



static int l_maxpoints(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	return mep->maxpoints(L);	
}

int MEP::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Calculate a minimum energy pathway between two states using an Elastic Band method.");
		//lua_pushstring(L, "1 *3Vector* or *SpinSystem*: System Size");
		lua_pushstring(L, "");
		lua_pushstring(L, ""); //output, empty
		return 3;
	}

	if(lua_istable(L, 1))
	{
		return 0;
	}

	lua_CFunction func = lua_tocfunction(L, 1);

	if(func == l_setdata)
	{
		lua_pushstring(L, "Set internal data. This method is used in the support lua scripts and should not be used.");
		lua_pushstring(L, "1 Value: the new internal data");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_getdata)
	{
		lua_pushstring(L, "Get internal data. This method is used in the support lua scripts and should not be used.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Value: the internal data");
		return 3;
	}
	if(func == l_getsite)
	{
		lua_pushstring(L, "Get site x,y and z coordinates.");
		lua_pushstring(L, "2 Integers: 1st integer is path index, 2nd integer is site index.");
		lua_pushstring(L, "4 Numbers: x,y,z,m orientation of spin and magnitude at site s at path point p.");
		return 3;
	}
	
	if(func == l_getallsites)
	{
		lua_pushstring(L, "Get which sites are involved in calculation");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Table of Tables of 3 Integers: Site positions involved in calculation.");
		return 3;
	}
	
	if(func == l_sitecount)
	{
		lua_pushstring(L, "Get number of sites invovled in the calculation");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: number of sites involved in the calculation.");
		return 3;
	}
	
	if(func == l_problemscale)
	{
		lua_pushstring(L, "Get the average magnitude of the moments involved in the calculation");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: Scale of the problem.");
		return 3;
	}
	
	return LuaBaseObject::help(L);
}



static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* MEP::luaMethods()
{
	if(m[127].name)return m;
	merge_luaL_Reg(m, LuaBaseObject::luaMethods());

	static const luaL_Reg _m[] =
	{
		{"setInternalData", l_setdata},
		{"getInternalData", l_getdata},
		{"_addSite", l_addsite},
		{"sites", l_getallsites},
		{"siteCount", l_sitecount},
		{"clearSites", l_clearsites},
		{"clearPath", l_clearpath},
		{"makeForcePerpendicularToSpins", l_projForcePerpSpins},
		{"makeForcePerpendicularToPath", l_projForcePerpPath},
		{"makeForceParallelToPath", l_projForcePath},
		{"calculateEnergyGradients", l_calculateEnergyGradients},
		{"resampleStateXYZPath", l_resampleStateXYZPath},
		{"getPathEnergy", l_getpathenergy},
		{"calculateEnergies", l_calculateEnergies},
		{"gradient", l_getgradient},
		{"spin", l_getsite},
		{"applyForces", l_applyforces},
		{"problemScale", l_problemscale},
		{"_addStateXYZ", l_addstatexyz},
		{"_relaxSinglePoint", l_relaxSinglePoint_}, //internal method
// 		{"_relaxSaddlePoint", l_relaxSaddlePoint_}, //internal method
		{"_hessianAtPoint", l_computepoint2deriv},
		{"_maximalPoints", l_maxpoints},
		{"_randomize", l_randomize_},

		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}






extern "C"
{
static int l_getmetatable(lua_State* L)
{
	if(!lua_isstring(L, 1))
		return luaL_error(L, "First argument must be a metatable name");
	luaL_getmetatable(L, lua_tostring(L, 1));
	return 1;
}

_MEP_API int lib_register(lua_State* L)
{
	luaT_register<MEP>(L);

	lua_pushcfunction(L, l_getmetatable);
	lua_setglobal(L, "maglua_getmetatable");
	if(luaL_dostring(L, __mep_luafuncs()))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		return luaL_error(L, lua_tostring(L, -1));
	}

	lua_pushnil(L);
	lua_setglobal(L, "maglua_getmetatable");


	return 0;
}

_MEP_API int lib_version(lua_State* L)
{
	return __revi;
}

_MEP_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "MEP";
#else
	return "MEP-Debug";
#endif
}

_MEP_API int lib_main(lua_State* L)
{
	return 0;
}
}
