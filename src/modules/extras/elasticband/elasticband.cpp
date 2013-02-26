#include "elasticband.h"
#include "elasticband_luafuncs.h"
#include "info.h"
#include "luamigrate.h"
#include <math.h>


// deterministic random number generator
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


ElasticBand::ElasticBand()
	: LuaBaseObject(hash32(ElasticBand::slineage(0)))
{
	ref_data = LUA_REFNIL;
}


ElasticBand::~ElasticBand()
{
	deinit();
}

void ElasticBand::init()
{
}

void ElasticBand::deinit()
{
	if(ref_data != LUA_REFNIL)
		luaL_unref(L, LUA_REGISTRYINDEX, ref_data);
	ref_data = LUA_REFNIL;

	state_xyz_path.clear();
	sites.clear();
}


int ElasticBand::luaInit(lua_State* L, int base)
{
	state_xyz_path.clear();
	sites.clear();
	
	return LuaBaseObject::luaInit(L, base);
}

void ElasticBand::encode(buffer* b)
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

int  ElasticBand::decode(buffer* b)
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


static double dot(const double* a, const double* b, int n = 3)
{
	double s = 0;
	for(int i=0; i<n; i++)
		s += a[i]*b[i];
	return s;
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
}

double ElasticBand::distanceBetweenPoints(int p1, int p2, int site)
{
	const int num_sites = sites.size() / 3;

	const double* v1 = &state_xyz_path[p1 * num_sites * 3 + site * 3];
	const double* v2 = &state_xyz_path[p2 * num_sites * 3 + site * 3];
	
	const double d = angleBetween(v1, v2);

	return d;
}

double ElasticBand::distanceBetweenHyperPoints(int p1, int p2)
{
	double d = 0;
	int n = sites.size() / 3;
	for(int i=0; i<n; i++)
	{
		d += distanceBetweenPoints(p1, p2, i);
	}
	return d;
}

void ElasticBand::interpolatePoints(const int p1, const int p2, const int site, const double ratio, vector<double>& dest, const double noise)
{
	const int num_sites = sites.size() / 3;
	const double* v1 = &state_xyz_path[p1 * num_sites * 3 + site * 3];
	const double* v2 = &state_xyz_path[p2 * num_sites * 3 + site * 3];
	
	double a = angleBetween(v1, v2);
	double norm[3];

	if(fabs(a - 3.1415926538979) < 1e-8) //then colinear, need a random ortho vector
	{
		double t[3]; 
		t[0] = -v2[2]; 
		t[1] = -v2[0]; 
		t[2] =  v2[1];
		cross(v1,t,norm);
		if(dot(norm, norm) == 0) //then
		{
			t[0] = myrandf()*2.0-1.0;
			t[1] = myrandf()*2.0-1.0;
			t[2] = myrandf()*2.0-1.0;
			cross(v1,t,norm); //assuming this works
		}
	}
	else
	{
		cross(v1,v2,norm);
	}

	// introducing noise in rotation vector! what a great idea
	if(noise != 0)
	{
		norm[0] += (myrandf()*2.0-1) * noise;
		norm[1] += (myrandf()*2.0-1) * noise;
		norm[2] += (myrandf()*2.0-1) * noise;
	}

	const double ln = sqrt(dot(norm,norm));
	
	// it is common to fall into this block when the
	// start and end points are the same
	if(ln == 0)
	{
		dest.push_back(v1[0]);
		dest.push_back(v1[1]);
		dest.push_back(v1[2]);
		return;
	}

	const double iln = 1.0/ln;
	norm[0] *= iln;
	norm[1] *= iln;
	norm[2] *= iln;
	
	
	double res[3];

	
	rotateAboutBy(v1, norm, -a*ratio, res);
	
	dest.push_back(res[0]);
	dest.push_back(res[1]);
	dest.push_back(res[2]);
}



void ElasticBand::interpolateHyperPoints(const int p1, const int p2, const double ratio, vector<double>& dest, const double noise)
{
	int n = sites.size() / 3;
	for(int i=0; i<n; i++)
	{
		interpolatePoints(p1,p2,i,ratio,dest,noise);
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

int ElasticBand::resampleStateXYZPath(lua_State* L, int new_num_points, const double noise)
{
	if(new_num_points < 2)
	{
		return luaL_error(L, "attempting to resample path to less than 2 points");
	}
	
	vector<double> new_state_xyz_path;
	const int num_points = sites.size() / 3;
	double distance = 0; //total distance

	vector<double> d12;
	
	const int num_hyperpoints = state_xyz_path.size()/(num_points * 3);
	
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
			interpolateHyperPoints(j, j+1, ratio, new_state_xyz_path, noise);
		}
	}
	
	// endpoints are fixed: path end
	const int last_set_idx = state_xyz_path.size() - sites.size();
	for(int i=0; i<sites.size(); i++)
	{
		//printf("last: %f\n", state_xyz_path[last_set_idx + i]);
		new_state_xyz_path.push_back( state_xyz_path[last_set_idx + i] );
	}
	
	state_xyz_path.clear();
	for(unsigned int i=0; i<new_state_xyz_path.size(); i++)
	{
		state_xyz_path.push_back(new_state_xyz_path[i]);
	}
}

void ElasticBand::addSite(int x, int y, int z)
{
	state_xyz_path.clear();
	sites.push_back(x);
	sites.push_back(y);
	sites.push_back(z);
}

void ElasticBand::projForcePerpSpins() //project gradients onto vector perpendicular to spin direction
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

void ElasticBand::calcSpringForces(double k)
{
	const int ss = sites.size();
	const int pp = state_xyz_path.size() / ss;

	double* tmp = new double[ss];

	force_vector.resize(state_xyz_path.size());
	for(int i=0; i<state_xyz_path.size(); i++)
	{
		force_vector[i] = 0;
	}

	for(int i=1; i<pp-1; i++)
	{
		{
			calculateOffsetVector(tmp, i-1, i);
			const double dist = sqrt(dot(tmp,tmp,ss));
			for(int j=0; j<ss; j++)
			{
				force_vector[i*ss + j] += k * dist * tmp[j];
			}
		}
		{	
			calculateOffsetVector(tmp, i+1, i);
			const double dist = sqrt(dot(tmp,tmp,ss));
			for(int j=0; j<ss; j++)
			{
				force_vector[i*ss + j] += k * dist * tmp[j];
			}
		}
	}

	delete [] tmp;

}

void ElasticBand::calculateOffsetVector(double* vec, const int p1, const int p2)
{
	const int ss = sites.size();
	for(int i=0; i<ss; i++)
	{
		vec[i] = state_xyz_path[p1*ss+i] - state_xyz_path[p2*ss+i];
	}
}


void ElasticBand::computeTangent(const int p1, const int p2, const int dest)
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


void ElasticBand::projForcePerpPath() //project gradients onto vector perpendicular to path direction
{
	path_tangent.clear();
	path_tangent.resize(state_xyz_path.size());

	const int s = sites.size()/3;

	computeTangent(0, 1, 0);
	for(int i=1; i<s-1; i++)
	{
		computeTangent(i-1,i+1,i);
	}
	computeTangent(s-2, s-1, s-1);



	const int ss = sites.size(); // point size
	const int nn = state_xyz_path.size()/sites.size(); // number of points

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



void ElasticBand::projForcePath() //project gradients onto vector perpendicular to path direction
{
	path_tangent.clear();
	path_tangent.resize(state_xyz_path.size());

	const int s = sites.size()/3;

	computeTangent(0, 1, 0);
	for(int i=1; i<s-1; i++)
	{
		computeTangent(i-1,i+1,i);
	}
	computeTangent(s-2, s-1, s-1);



	const int ss = sites.size(); // point size
	const int nn = state_xyz_path.size()/sites.size(); // number of points

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


// expected on the stack:
// at 1, elasticband
// at 2, function get_site_ss1(x,y,z) return sx,sy,sz end
// at 3, function get_site_ss2(x,y,z) return sx,sy,sz end
int ElasticBand::initializeEndpoints(lua_State* L)
{
	state_xyz_path.clear();
	state_xyz_path.resize( sites.size() * 2 );
	for(int p=0; p<2; p++)
	{
		for(int s=0; s<sites.size(); s+=3)
		{
			lua_pushvalue(L, 2+p);
			lua_pushinteger(L, sites[s+0]+1);
			lua_pushinteger(L, sites[s+1]+1);
			lua_pushinteger(L, sites[s+2]+1);
			lua_call(L, 3, 4);
			int p1 = sites.size()*p + s + 0;
			int p2 = sites.size()*p + s + 1;
			int p3 = sites.size()*p + s + 2;

			double v1 = lua_tonumber(L, -4);
			double v2 = lua_tonumber(L, -3);
			double v3 = lua_tonumber(L, -2);

			state_xyz_path[p1] = v1;
			state_xyz_path[p2] = v2;
			state_xyz_path[p3] = v3;

			lua_pop(L, 4);
		}
	}

	return 0;
}


// expected on the stack:
// at 1, elasticband
// at 2, function get_site_ss1(x,y,z) return sx,sy,sz,m
// at 3, function set_site_ss1(x,y,z,m  sx,sy,sz) 
// at 4, function get_energy_ss1()
int ElasticBand::calculateEnergies(lua_State* L)
{
	energies.clear();
	const int num_sites = sites.size() / 3;
	const int path_length = state_xyz_path.size() / sites.size();

	// back up old sites so we can restore after
	vector<double> old_sites;
	for(int s=0; s<num_sites; s++)
	{
		lua_pushvalue(L, 2);
		lua_pushinteger(L, sites[s*3+0]+1);
		lua_pushinteger(L, sites[s*3+1]+1);
		lua_pushinteger(L, sites[s*3+2]+1);
		lua_call(L, 3, 4);

		old_sites.push_back(lua_tonumber(L, -4));
		old_sites.push_back(lua_tonumber(L, -3));
		old_sites.push_back(lua_tonumber(L, -2));
		old_sites.push_back(lua_tonumber(L, -1));
		lua_pop(L, 4);
	}


	for(int p=0; p<path_length; p++)
	{
		double e = 0;
		for(int s=0; s<num_sites; s++)
		{
			lua_pushvalue(L, 3);
			lua_pushinteger(L, sites[s*3+0]+1);
			lua_pushinteger(L, sites[s*3+1]+1);
			lua_pushinteger(L, sites[s*3+2]+1);
			const double xx = state_xyz_path[p * sites.size() + s*3 + 0];
			const double yy = state_xyz_path[p * sites.size() + s*3 + 1];
			const double zz = state_xyz_path[p * sites.size() + s*3 + 2];
			lua_pushnumber(L, xx);
			lua_pushnumber(L, yy);
			lua_pushnumber(L, zz);
			lua_pushnumber(L, sqrt(xx*xx + yy*yy + zz*zz));
			lua_call(L, 7, 0);
		}

		lua_pushvalue(L, 4);
		lua_call(L, 0, 1);
		
		const double eps = lua_tonumber(L, -1);
		lua_pop(L, 1);
		e += eps;
		
		energies.push_back(e);
	}

	// restore state
	for(int s=0; s<num_sites; s++)
	{
		lua_pushvalue(L, 3);
		lua_pushinteger(L, sites[s*3+0]+1);
		lua_pushinteger(L, sites[s*3+1]+1);
		lua_pushinteger(L, sites[s*3+2]+1);
		lua_pushnumber(L, old_sites[s*4+0]);
		lua_pushnumber(L, old_sites[s*4+1]);
		lua_pushnumber(L, old_sites[s*4+2]);
		lua_pushnumber(L, old_sites[s*4+3]);
		lua_call(L, 7, 0);
	}
}


// expected on the stack:
// at 1, elasticband
// at 2, function get_site_ss1(x,y,z) return sx,sy,sz 
// at 3, function set_site_ss1(x,y,z, sx,sy,sz) 
// at 4, function get_energy_ss1()
// at 5, epsilon
int ElasticBand::calculateEnergyGradients(lua_State* L)
{
	const int num_sites = sites.size() / 3;
	const int path_length = state_xyz_path.size() / sites.size();
	const double epsilon = lua_tonumber(L, 5);
	
	force_vector.resize( state_xyz_path.size() ); //since we're doing random-ish access
	
	if(epsilon == 0)
		return luaL_error(L, "epsilon is zero in finite difference");
	
// 	double oxyz[4]; //original values x,y,z,m
	
	// need to save current configuration
	vector<double> oxyzm;
	for(int s=0; s<num_sites; s++)
	{
		lua_pushvalue(L, 2);
		lua_pushinteger(L, sites[s*3+0]+1);
		lua_pushinteger(L, sites[s*3+1]+1);
		lua_pushinteger(L, sites[s*3+2]+1);
		lua_call(L, 3, 4);
		
		oxyzm.push_back(lua_tonumber(L, -4));
		oxyzm.push_back(lua_tonumber(L, -3));
		oxyzm.push_back(lua_tonumber(L, -2));
		oxyzm.push_back(lua_tonumber(L, -1));
		lua_pop(L, 4);
	}
	
	// lets march along the path
	for(int p=0; p<path_length; p++)
	{
		double* vxyz = &state_xyz_path[p*num_sites*3];
		
		// write path point as it currently stands
		for(int s=0; s<num_sites; s++)
		{
			lua_pushvalue(L, 3);
			lua_pushinteger(L, sites[s*3+0]+1);
			lua_pushinteger(L, sites[s*3+1]+1);
			lua_pushinteger(L, sites[s*3+2]+1);
				
			lua_pushnumber(L, state_xyz_path[p*num_sites*3 + s*3 + 0]);
			lua_pushnumber(L, state_xyz_path[p*num_sites*3 + s*3 + 1]);
			lua_pushnumber(L, state_xyz_path[p*num_sites*3 + s*3 + 2]);
			lua_pushnumber(L, oxyzm[s*4+3]);
			lua_call(L, 7, 0);
		}
		
		for(int s=0; s<num_sites; s++)
		{
			double* vxyz = &state_xyz_path[p*num_sites*3 + s*3];

			// We now have a point vxyz at site p,s
			// time to compute the spatial energy gradient.
			double g[6];
			
			
			// 7th below is to restore old state
			const double cc[7][3] = {{1,0,0}, {-1,0,0},   {0,1,0},{0,-1,0},   {0,0,1},{0,0,-1}, {0,0,0}};
			for(int c=0; c<7; c++)
			{
				const double new_mx = vxyz[0] + cc[c][0] * epsilon;
				const double new_my = vxyz[1] + cc[c][1] * epsilon;
				const double new_mz = vxyz[2] + cc[c][2] * epsilon;
				
				lua_pushvalue(L, 3);
				lua_pushinteger(L, sites[s*3+0]+1);
				lua_pushinteger(L, sites[s*3+1]+1);
				lua_pushinteger(L, sites[s*3+2]+1);
				
				lua_pushnumber(L, new_mx);
				lua_pushnumber(L, new_my);
				lua_pushnumber(L, new_mz);
				lua_pushnumber(L, oxyzm[s*4+3]);
				lua_call(L, 7, 0);
				
				if(c < 6) 
				{
					lua_pushvalue(L, 4); //get energy
					lua_call(L, 0, 1);
					
					g[c] = lua_tonumber(L, -1);
					
// 					printf("(%i %i %i)  (%f %f %f:%f) = %f\n", 
// 					sites[s*3+0]+1, sites[s*3+1]+1, sites[s*3+2]+1,
// 					new_mx, new_my, new_mz,oxyz[3], g[c]);
	
					lua_pop(L, 1);
				}
			}
			
			// now have 6 energy samples for this point p at site s
// 			printf("writing to %i of %i\n", p*num_sites*3 + s*3 + 0, state_xyz_path.size());
			force_vector[p*num_sites*3 + s*3 + 0] = (g[1] - g[0]) / (2.0 * epsilon);
			force_vector[p*num_sites*3 + s*3 + 1] = (g[3] - g[2]) / (2.0 * epsilon);
			force_vector[p*num_sites*3 + s*3 + 2] = (g[5] - g[4]) / (2.0 * epsilon);
		}
		
	}
	
	// need to restore saved configuration
	int j = 0;
	for(int s=0; s<num_sites; s++)
	{
		// now to restore things back to how they were
		lua_pushvalue(L, 3);
		lua_pushinteger(L, sites[s*3+0]+1);
		lua_pushinteger(L, sites[s*3+1]+1);
		lua_pushinteger(L, sites[s*3+2]+1);	
		
		lua_pushnumber(L, oxyzm[j]); j++;
		lua_pushnumber(L, oxyzm[j]); j++;
		lua_pushnumber(L, oxyzm[j]); j++;
		lua_pushnumber(L, oxyzm[j]); j++;
		lua_call(L, 7, 0);
	}
	
	return 0;
}


// should move this to a class method
static int l_applyforces(lua_State* L)
{
	LUA_PREAMBLE(ElasticBand, eb, 1);

	double dt = 0.1;
	if(lua_isnumber(L, 2))
	{
		dt = lua_tonumber(L, 2);
	}
	
	if(eb->force_vector.size() < eb->state_xyz_path.size())
		return luaL_error(L, "forces are not computed or size mismatch");
	
	// end points are fixed
	for(int i=0; i<(eb->force_vector.size()/3); i++)
	{
		double x = eb->state_xyz_path[i*3+0];
		double y = eb->state_xyz_path[i*3+1];
		double z = eb->state_xyz_path[i*3+2];
		
		const double m = sqrt(x*x+y*y+z*z);
		
		x += eb->force_vector[i*3+0] * dt;
		y += eb->force_vector[i*3+1] * dt;
		z += eb->force_vector[i*3+2] * dt;
		
		const double m2 = sqrt(x*x+y*y+z*z);

		if(m2 > 0)
		{
			x *= m/m2;
			y *= m/m2;
			z *= m/m2;
		}
		
		eb->state_xyz_path[i*3+0] = x;
		eb->state_xyz_path[i*3+1] = y;
		eb->state_xyz_path[i*3+2] = z;
	}
	
	return 0;	
}







static int l_setdata(lua_State* L)
{
	LUA_PREAMBLE(ElasticBand, eb, 1);
	luaL_unref(L, LUA_REGISTRYINDEX, eb->ref_data);
	lua_pushvalue(L, 2);
	eb->ref_data = luaL_ref(L, LUA_REGISTRYINDEX);
	return 0;
}

static int l_getdata(lua_State* L)
{
	LUA_PREAMBLE(ElasticBand, eb, 1);
	lua_rawgeti(L, LUA_REGISTRYINDEX, eb->ref_data);
	return 1;
}


static int l_addsite(lua_State* L)
{
	LUA_PREAMBLE(ElasticBand, eb, 1);
	
	int site[3] = {0,0,0};
	
	if(lua_istable(L, 2))
	{
		for(int i=0; i<3; i++)
		{
			lua_pushinteger(L, i+1);
			lua_gettable(L, 2);
			if(lua_isnumber(L, -1))
			{
				site[i] = lua_tointeger(L, -1) - 1;
			}
			lua_pop(L, 1);
		}
	}
	else
	{
		for(int i=0; i<3; i++)
		{
			if(lua_isnumber(L, 2+i))
			{
				site[i] = lua_tointeger(L, 2+i) - 1;
			}
		}
	}
	
	eb->addSite(site[0], site[1], site[2]);
}

static int l_clearsites(lua_State* L)
{
	LUA_PREAMBLE(ElasticBand, eb, 1);
	eb->sites.clear();
	eb->state_xyz_path.clear();
	return 0;
}

static int l_getallsites(lua_State* L)
{
	LUA_PREAMBLE(ElasticBand, eb, 1);

	lua_newtable(L);
	
	for(unsigned int i=0; i<eb->sites.size()/3; i++)
	{
		lua_pushinteger(L, i+1);
		lua_newtable(L);
		for(int j=0; j<3; j++)
		{
			lua_pushinteger(L, j+1);
			lua_pushinteger(L, eb->sites[i*3+j]+1);
			lua_settable(L, -3);
		}
		lua_settable(L, -3);
	} 	
	return 1;
}

static int l_initializeEndpoints(lua_State* L)
{
	LUA_PREAMBLE(ElasticBand, eb, 1);

	if(!lua_isfunction(L, 2))
		return luaL_error(L, "1st argument expected to be get_site_ss1(x,y,z) function");
	if(!lua_isfunction(L, 3))
		return luaL_error(L, "2nd argument expected to be get_site_ss2(x,y,z) function");
	return eb->initializeEndpoints(L);
}

static int l_resampleStateXYZPath(lua_State* L)
{
	LUA_PREAMBLE(ElasticBand, eb, 1);

	const int n = lua_tointeger(L, 2);
	double noise = 0;
	
	if(lua_isnumber(L, 3))
	{
		noise = lua_tonumber(L, 3);
	}
	
	return eb->resampleStateXYZPath(L, n, noise);
}

static int l_calcspringforce(lua_State* L)
{
    LUA_PREAMBLE(ElasticBand, eb, 1);
	double k = 1;
	if(lua_isnumber(L, 2))
		k = lua_tonumber(L, 2);
	eb->calcSpringForces(k);
	return 0;
}

static int l_projForcePerpSpins(lua_State* L)
{
	LUA_PREAMBLE(ElasticBand, eb, 1);
	eb->projForcePerpSpins();
	return 0;
}
static int l_projForcePerpPath(lua_State* L)
{
	LUA_PREAMBLE(ElasticBand, eb, 1);
	eb->projForcePerpPath();
	return 0;
}
static int l_projForcePath(lua_State* L)
{
	LUA_PREAMBLE(ElasticBand, eb, 1);
	eb->projForcePath();
	return 0;
}

// at 1, elasticband
// at 2, function get_site(x,y,z) return sx,sy,sz end
// at 3, function set_site(x,y,z, sx,sy,sz) end
// at 4, get energy function
// at 5, epsilon
static int l_calculateEnergyGradients(lua_State* L)
{
	LUA_PREAMBLE(ElasticBand, eb, 1);
	
	if(!lua_isfunction(L, 2))
		return luaL_error(L, "1st argument expected to be get_site_ss1(x,y,z) function");
	if(!lua_isfunction(L, 3))
		return luaL_error(L, "2nd argument expected to be set_site_ss1(x,y,z,sx,sy,sz) function");
	if(!lua_isfunction(L, 4))
		return luaL_error(L, "3rd argument expected to be get_energy_ss1() function");
	if(!lua_isnumber(L, 5))
		return luaL_error(L, "4th argument expected to be epsilon for finite difference");
	
	return eb->calculateEnergyGradients(L);
}

// at 1, elasticband
// at 2, function get_site(x,y,z) return sx,sy,sz end
// at 3, function set_site(x,y,z, sx,sy,sz) end
// at 4, get energy function
static int l_calculateEnergies(lua_State* L)
{
	LUA_PREAMBLE(ElasticBand, eb, 1);
	
	if(!lua_isfunction(L, 2))
		return luaL_error(L, "1st argument expected to be get_site_ss1(x,y,z) function");
	if(!lua_isfunction(L, 3))
		return luaL_error(L, "2nd argument expected to be set_site_ss1(x,y,z,sx,sy,sz) function");
	if(!lua_isfunction(L, 4))
		return luaL_error(L, "3rd argument expected to be get_energy_ss1() function");
	
	return eb->calculateEnergies(L);
}

static int l_getpathenergy(lua_State* L)
{
    LUA_PREAMBLE(ElasticBand, eb, 1);

	lua_newtable(L);
	for(int i=0; i<eb->energies.size(); i++)
	{
		lua_pushinteger(L, i+1);
		lua_pushnumber(L, eb->energies[i]);
		lua_settable(L, -3);
	}
	return 1;
}

//eb->sites.clear();
//eb->state_xyz_path.clear();
static int l_getgradient(lua_State* L)
{
    LUA_PREAMBLE(ElasticBand, eb, 1);

	const int p = lua_tointeger(L, 2) - 1;
	const int s = lua_tointeger(L, 3) - 1;

	if(p < 0 || s < 0)
		return 0;

	int idx = eb->sites.size() * p + s*3;

	if(eb->force_vector.size() < idx+2)
		return 0;

	lua_pushnumber(L, eb->force_vector[idx+0]);
	lua_pushnumber(L, eb->force_vector[idx+1]);
	lua_pushnumber(L, eb->force_vector[idx+2]);
	return 3;
}

static int l_getsite(lua_State* L)
{
    LUA_PREAMBLE(ElasticBand, eb, 1);

	const int p = lua_tointeger(L, 2) - 1;
	const int s = lua_tointeger(L, 3) - 1;

	if(p < 0 || s < 0)
	{
		return luaL_error(L, "Site or point is out of bounds");
	}

	int idx = eb->sites.size() * p + s*3;

	if(eb->state_xyz_path.size() < idx+2)
		return luaL_error(L, "Site or point is out of bounds");

	lua_pushnumber(L, eb->state_xyz_path[idx+0]);
	lua_pushnumber(L, eb->state_xyz_path[idx+1]);
	lua_pushnumber(L, eb->state_xyz_path[idx+2]);
	return 3;
}



int ElasticBand::help(lua_State* L)
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
		lua_pushstring(L, "3 Numbers: x,y,z coordinates of spin at site s at path point p.");
		return 3;
	}
	
	if(func == l_getallsites)
	{
		lua_pushstring(L, "Get which sites are involved in calculation");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Table of Tables of 3 Integers: Site positions involved in calculation.");
		return 3;
	}
	return LuaBaseObject::help(L);
}



static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* ElasticBand::luaMethods()
{
	if(m[127].name)return m;
	merge_luaL_Reg(m, LuaBaseObject::luaMethods());

	static const luaL_Reg _m[] =
	{
		{"setInternalData", l_setdata},
		{"getInternalData", l_getdata},
		{"addSite", l_addsite},
		{"sites", l_getallsites},
		{"clearSites", l_clearsites},
		{"makeForcePerpendicularToSpins", l_projForcePerpSpins},
		{"makeForcePerpendicularToPath", l_projForcePerpPath},
		{"makeForceParallelToPath", l_projForcePath},
		{"calculateEnergyGradients", l_calculateEnergyGradients},
		{"calculateSpringForces", l_calcspringforce},
		{"initializeEndpoints", l_initializeEndpoints},
		{"resampleStateXYZPath", l_resampleStateXYZPath},
		{"getPathEnergy", l_getpathenergy},
		{"calculateEnergies", l_calculateEnergies},
		{"gradient", l_getgradient},
		{"spin", l_getsite},
		{"applyForces", l_applyforces},
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

ELASTICBAND_API int lib_register(lua_State* L)
{
	luaT_register<ElasticBand>(L);

	lua_pushcfunction(L, l_getmetatable);
	lua_setglobal(L, "maglua_getmetatable");
	if(luaL_dostring(L, __elasticband_luafuncs()))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		return luaL_error(L, lua_tostring(L, -1));
	}

	lua_pushnil(L);
	lua_setglobal(L, "maglua_getmetatable");


	return 0;
}

ELASTICBAND_API int lib_version(lua_State* L)
{
	return __revi;
}

ELASTICBAND_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "ElasticBand";
#else
	return "ElasticBand-Debug";
#endif
}

ELASTICBAND_API int lib_main(lua_State* L)
{
	return 0;
}
}
