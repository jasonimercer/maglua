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

static void normalizeTo(double* dest, const double* src, const double length=1.0, const int n=3)
{
	double len = dot(src, src, n);
	if(len == 0)
	{
		dest[0] = length;
		for(int i=1; i<n; i++)
			dest[i] = 0;
	}
	else
	{
		const double iln = length/sqrt(len);
		for(int i=0; i<n; i++)
			dest[i] = iln * src[i];
	}
}

static void project(double* dest, const double* a, const double* b, int n=3)
{
	double ab = dot(a,b,n);
	double bb = dot(b,b,n);
	
	if(bb == 0)
		normalizeTo(dest, b, 0, n);
	else
		normalizeTo(dest, b, ab/bb, n);
}

static void reject(double* dest, const double* a, const double* b, int n=3)
{
	project(dest, a, b, n);
	for(int i=0; i<n; i++)
		dest[i] = a[i] - dest[i];
}


static double angleBetween(const double* a, const double* b)
{
	const double n = sqrt(dot(a,a) * dot(b,b));
	
// 	printf("a: %g %g %g\n", a[0], a[1], a[2]);
// 	printf("b: %g %g %g\n", b[0], b[1], b[2]);
	
	if(n == 0)
		return 0;

	double ct = dot(a,b) / n;
	if(ct < -1)
		ct = -1;
	if(ct > 1)
		ct = 1;
	const double acos_ct = acos(ct);
// 	printf("n: %g   ct: %g    acos_ct: %g\n", n, ct, acos_ct);
	return acos_ct;
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
	good_distances = false;
	beta = 0.1;
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
	image_site_mobility.clear();
	sites.clear();
}


int MEP::luaInit(lua_State* L, int base)
{
	state_xyz_path.clear();
	image_site_mobility.clear();
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
	
	encodeInteger(image_site_mobility.size(), b);
	for(int i=0; i<image_site_mobility.size(); i++)
	{
		encodeDouble(image_site_mobility[i], b);
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
		image_site_mobility.push_back(decodeDouble(b));
	}
	n = decodeInteger(b);
	for(int i=0; i<n; i++)
	{
		sites.push_back(decodeInteger(b));
	}
	
	return 0;
}

void MEP::internal_copy_to(MEP* dest)
{
	dest->state_xyz_path.clear();
	for(int i=0; i<state_xyz_path.size(); i++)
	{
		dest->state_xyz_path.push_back(state_xyz_path[i]);
	}
	
	dest->image_site_mobility.clear();
	for(int i=0; i<image_site_mobility.size(); i++)
	{
		dest->image_site_mobility.push_back(image_site_mobility[i]);
	}

	//dest->state_xyz_path = state_xyz_path;
	dest->sites = sites;
	dest->path_tangent = path_tangent;
	dest->force_vector = force_vector;
	dest->energies = energies;
	dest->energy_ok = false;
	dest->beta = beta;
}

double MEP::absoluteDifference(MEP* other, int point, double& max_diff)
{
	if(state_xyz_path.size() != other->state_xyz_path.size())
	{
// 		printf("s1.size() = %i, s2.size() = %i\n", state_xyz_path.size(), other->state_xyz_path.size());
		return -1;
	}

	double d = 0;
	max_diff = 0;

	double n1[3];
	double n2[3];
	double v[3];

	int min = 0;
	int max = state_xyz_path.size();

	if(point >= 0)
	{
		min =  point    * numberOfSites() * 3;
		max = (point+1) * numberOfSites() * 3;
	}

	for(int i=min; i<max; i+=3)
	{
		normalizeTo(n1, &(state_xyz_path[i]));
		normalizeTo(n2, &(other->state_xyz_path[i]));

		v[0] = n1[0] - n2[0];
		v[1] = n1[1] - n2[1];
		v[2] = n1[2] - n2[2];

		const double diff = sqrt(dot(v,v));

		if(diff > max_diff)
			max_diff = diff;
		
		d += diff;
	}

	return d;
}



void MEP::randomize(const double magnitude)
{
	double* v = &state_xyz_path[0];

	const int num_sites = numberOfSites();
	double nv[3];

	const int ni = numberOfImages(); //images
	const int ns = numberOfSites();
	
	for(int i=0; i<ni; i++)
	{
		for(int s=0; s<ns; s++)
		{
			const int k = (i*ns+s);
			
			double x = v[k*3+0];
			double y = v[k*3+1];
			double z = v[k*3+2];
			
			const double m1 = sqrt(x*x + y*y + z*z);
			
			normalizeTo(nv, &(v[k*3]));
			
			nv[0] += (myrandf() * magnitude * image_site_mobility[k]);
			nv[1] += (myrandf() * magnitude * image_site_mobility[k]);
			nv[2] += (myrandf() * magnitude * image_site_mobility[k]);
			
			normalizeTo(&(v[k*3]), nv, m1);
		}
	}
}



double MEP::distanceBetweenPoints(int p1, int p2, int site)
{
	const int num_sites = sites.size() / 3;

	const double* v1 = &state_xyz_path[p1 * num_sites * 3 + site * 3];
	const double* v2 = &state_xyz_path[p2 * num_sites * 3 + site * 3];
	
	const double d = angleBetween(v1, v2);

// 	printf("v1: %g %g %g\n", v1[0], v1[1], v1[2]);
// 	printf("v2: %g %g %g\n", v2[0], v2[1], v2[2]);
// 	printf("abp %i %i %g\n", p1, p2,  d);
	
	return d;
}

double MEP::distanceBetweenHyperPoints(int p1, int p2)
{
	double d = 0;
	int n = sites.size() / 3;
	for(int i=0; i<n; i++)
	{
		const double dbp = distanceBetweenPoints(p1, p2, i);
		d += dbp;
	}
	return d;
}

void MEP::interpolatePoints(const int p1, const int p2, const int site, const double _ratio, vector<double>& dest, const double rjitter)
{
	const int num_sites = sites.size() / 3;
	const double* v1 = &state_xyz_path[p1 * num_sites * 3 + site * 3];
	const double* v2 = &state_xyz_path[p2 * num_sites * 3 + site * 3];
	
	double nv1[3];
	double nv2[3];

	double mobility = 1.0;
	
	const double m1 = getImageSiteMobility(p1, site);
	const double m2 = getImageSiteMobility(p2, site);
	
	if(m1 < mobility)
		mobility = m1;
	if(m2 < mobility)
		mobility = m2;
	
	normalizeTo(nv1, v1);
	normalizeTo(nv2, v2);
	
// 	double rjitter = (myrandf() * jitter*2 - jitter) * mobility;

	double ratio = _ratio * (1.0 + rjitter * mobility);
	
	double a = angleBetween(nv1, nv2);
	
	if(a == 0 || ratio == 0) //then  no need to interpolate
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



void MEP::interpolateHyperPoints(const int p1, const int p2, const double ratio, vector<double>& dest, const double jitter)
{
	int n = numberOfSites();
	//printf("(%s:%i) %i\n", __FILE__, __LINE__, n);
	for(int i=0; i<n; i++)
	{
		double rjitter = (myrandf() * jitter*2 - jitter);
		interpolatePoints(p1,p2,i,ratio,dest, rjitter);
	}
}


// get interval that bounds the distance
static int get_interval(const vector<double>& v, double x, double& ratio)
{
// 	printf("x = %f\n", x);	
	for(unsigned int i=0; i<v.size(); i++)
	{
		const double y = v[i];
// 		printf(">> %i %g %g\n", i,x,y);
		if(x < y)
		{
// 			printf("y: %g\n");
			ratio = x/y;
// 			printf("ratio: %g\n", ratio);
			return i;
		}
		x -= y;
	}
	
	// clip to end
	ratio = 1;
	return v.size()-1;
	
// 	return -1;
}

void MEP::printState()
{
	printf("State Path\n");
	
	for(int i=0; i<state_xyz_path.size(); i+=3)
	{
		const int k = i/3;
		printf("%3i\t%g\t%g\t%g   m=%g\n", k, state_xyz_path[i+0], state_xyz_path[i+1],state_xyz_path[i+2], image_site_mobility[k]);
	}
}

double MEP::calculateD12()
{
	if(good_distances)
		return d12_total;

	d12.clear();
	const int num_hyperpoints = numberOfImages();
	d12_total = 0;
	for(int i=0; i<num_hyperpoints-1; i++)
	{
		const double d = distanceBetweenHyperPoints(i,i+1);
		d12.push_back(d);
		d12_total += d;
	}
	
	good_distances = true;
	return d12_total;
}

#include <algorithm>    // std::sort
int MEP::resampleStateXYZPath(lua_State* L)
{
	vector<double> points;
	
	double jitter = 0; //percent
	if(lua_isnumber(L, 3))
		jitter = lua_tonumber(L, 3);
	
	if(lua_isnumber(L, 2))
	{
		int num_points = lua_tointeger(L, 2);
		if(num_points < 2)
			num_points = 2;
		
		for(int i=0; i<num_points; i++)
			points.push_back(i);
	}
	else
	{
		if(lua_istable(L, 2))
		{
			lua_pushnil(L);  /* first key */
			while(lua_next(L, 2) != 0)
			{
				points.push_back(lua_tonumber(L, -1));
				lua_pop(L, 1);
			}
		}
		else
		{
			return luaL_error(L, "resampleStateXYZPath requires a number of a table of numbers");
		}
	}
	
	
	std::sort(points.begin(), points.end());

	if(lua_isnumber(L, 4))
	{
		int goal_pts = lua_tointeger(L, 4);
		
		int current_pts = points.size();
		
		int magic_number = current_pts * 2 - 2; //this is the number of bases+offsets
		// we'll grow in the positive direction from the 1st number, 
		// positive and negative directions from middle numbers
		// and negative direction from the end number
		
		double* base_offsets = new double[magic_number * 2];
		int j = 0;
		base_offsets[j+0] = points[0];
		base_offsets[j+1] = 0.25 * (points[1] - points[0]);
		
		j += 2;
		for(int i=1; i<current_pts-1; i++)
		{
			base_offsets[j+0]   = points[i];
			base_offsets[j+1] = 0.1 * (points[i-1] - points[i]);
			base_offsets[j+2]   = points[i];
			base_offsets[j+3] = 0.25 * (points[i+1] - points[i]);
			j += 4;
		}
		
		base_offsets[j+0]   = points[current_pts-1];
		base_offsets[j+1] = 0.25 * (points[current_pts-2] - points[current_pts-1]);
		
		j = 0;
		while(points.size() < goal_pts)
		{
			points.push_back( base_offsets[j*2+0] + base_offsets[j*2+1] );
			base_offsets[j*2+1] *= 0.5;
			j++;
			
			if(j >= magic_number)
				j = 0;
		}
		
		
		
		delete [] base_offsets;

		std::sort(points.begin(), points.end());
	}
	
	
	const double max_point = points[ points.size() - 1];
	double distance = calculateD12(); //total distance
	vector<double> new_state_xyz_path;

	int new_num_images = points.size();
	// scaling points to same range as current points
	for(int i=0; i<new_num_images; i++)
	{
		points[i] *= (distance/max_point);
	}
	
	double ratio;
	for(int i=0; i<points.size(); i++)
	{
		int j = get_interval(d12, points[i], ratio);
		if(j != -1)
		{
// 			printf("range[ %i:%i]  ratio: %e\n", j, j+1, ratio); 
			interpolateHyperPoints(j, j+1, ratio, new_state_xyz_path, jitter);
		}
		else
		{
			printf("failed to get interval\n");
			printf("  total distance: %g\n", distance);
			printf("  requested position: %g\n", points[i]);
		}
	}
	
	state_xyz_path.clear();
	for(unsigned int i=0; i<new_state_xyz_path.size(); i++)
	{
		state_xyz_path.push_back(new_state_xyz_path[i]);
	}
	
	// now we need to update mobility factors. The only case we will cover
	// is non-unitary endpoints.
	vector<double> first_mobility;// = image_site_mobility[0];
	vector<double> last_mobility;// = image_site_mobility[image_site_mobility.size()-1];
	
	const int ns = numberOfSites();
// 	new_num_images
	
	for(int i=0; i<ns; i++)
	{
		first_mobility.push_back( image_site_mobility[i] );
	}

	for(int i=image_site_mobility.size()-ns; i<image_site_mobility.size(); i++)
	{
		last_mobility.push_back( image_site_mobility[i] ); 
	}
	
	image_site_mobility.clear();
	
	for(int i=0; i<ns; i++)
	{
		image_site_mobility.push_back(first_mobility[i]);
	}
	
	
	for(int i=0; i<ns*(new_num_images-2); i++)
	{
		image_site_mobility.push_back(1.0);
	}
	
	for(int i=0; i<ns; i++)
	{
		image_site_mobility.push_back(last_mobility[i]);
	}
	
	energy_ok = false; //need to recalc energy
	good_distances = false;
	return 0;
}

#if 0
int MEP::resampleStateXYZPath(lua_State* L)
{
//  	printf("resample from %i\n", state_xyz_path.size());

	int new_num_images = lua_tointeger(L, 2);
	
	double 
	
	double jitter = 0; //percent
	if(lua_isnumber(L, 3))
		jitter = lua_tonumber(L, 3);

	if(new_num_images < 2)
	{
		return luaL_error(L, "attempting to resample path to less than 2 images");
	}
	
// 	printf("pre-resample\n");
// 	printState();
	
	vector<double> new_state_xyz_path;
	const int num_points = numberOfSites();
	double distance = calculateD12(); //total distance


	
	// endpoints are fixed for resample path start
// 	for(int i=0; i<sites.size(); i++)
// 	{
// 		new_state_xyz_path.push_back( state_xyz_path[i] );
// 	}
// 	printf("distance = %f\n", distance);
	const double interval = distance / ((double)(new_num_images-1));
	double ratio;
	
	for(int i=0; i<new_num_images; i++)
	{
		int j = get_interval(d12, interval * i, ratio);
		if(j != -1)
		{
// 			printf("range[ %i:%i]  ratio: %e\n", j, j+1, ratio); 
			interpolateHyperPoints(j, j+1, ratio, new_state_xyz_path, jitter);
		}
		else
		{
			printf("failed to get interval\n");
			printf("  total distance: %g\n", distance);
			printf("  requested position: %g\n", interval * i);
		}
	}
	
	// endpoints are fixed for resample path end
// 	const int last_set_idx = state_xyz_path.size() - sites.size();
// 	for(int i=0; i<sites.size(); i++)
// 	{
// 		//printf("last: %f\n", state_xyz_path[last_set_idx + i]);
// 		new_state_xyz_path.push_back( state_xyz_path[last_set_idx + i] );
// 	}
	
	
	state_xyz_path.clear();
	for(unsigned int i=0; i<new_state_xyz_path.size(); i++)
	{
		state_xyz_path.push_back(new_state_xyz_path[i]);
	}
	
	// now we need to update mobility factors. The only case we will cover
	// is non-unitary endpoints.
	vector<double> first_mobility;// = image_site_mobility[0];
	vector<double> last_mobility;// = image_site_mobility[image_site_mobility.size()-1];
	
	const int ns = numberOfSites();
// 	new_num_images
	
	for(int i=0; i<ns; i++)
	{
		first_mobility.push_back( image_site_mobility[i] );
	}

	for(int i=image_site_mobility.size()-ns; i<image_site_mobility.size(); i++)
	{
		last_mobility.push_back( image_site_mobility[i] ); 
	}
	
	image_site_mobility.clear();
	
	for(int i=0; i<ns; i++)
	{
		image_site_mobility.push_back(first_mobility[i]);
	}
	
	
	for(int i=0; i<ns*(new_num_images-2); i++)
	{
		image_site_mobility.push_back(1.0);
	}
	
	for(int i=0; i<ns; i++)
	{
		image_site_mobility.push_back(last_mobility[i]);
	}
	
	energy_ok = false; //need to recalc energy
	good_distances = false;
	
// 	printf("post-resample\n");
// 	printState();

// 	printf("resample to %i new_images -- (%i)\n", new_num_images, state_xyz_path.size() / (numberOfSites() * 3));
	
	return 0;
}
#endif


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

int MEP::applyForces(lua_State* L)
{
	if(force_vector.size() != state_xyz_path.size())
		return luaL_error(L, "forces are not computed or size mismatch");
	
	const int num_sites = numberOfSites();
	double absMovement = 0;
	double maxMovement = 0;
	
	const int ni = numberOfImages();
	const int ns = numberOfSites();
	
	// end points are no longer fixed. 
	//for(int i=num_sites; i<(force_vector.size()/3)-num_sites; i++)
	for(int i=0; i<ni; i++)
	{
		for(int s=0; s<ns; s++)
		{
			const int k = (i*ns+s);

			double x = state_xyz_path[k*3+0];
			double y = state_xyz_path[k*3+1];
			double z = state_xyz_path[k*3+2];
			
			const double m1 = sqrt(x*x+y*y+z*z);
			
			const double dx = force_vector[k*3+0] * image_site_mobility[k];
			const double dy = force_vector[k*3+1] * image_site_mobility[k];
			const double dz = force_vector[k*3+2] * image_site_mobility[k];
			
			x += dx;
			y += dy;
			z += dz;
			
			const double dd = sqrt(dx*dx + dy*dy + dz*dz);
			const double normalized_movement = dd/m1;

			absMovement += normalized_movement;
			if(maxMovement < normalized_movement)
				maxMovement = normalized_movement;

// 			printf("ssss: %i %i %g\n", image_site_mobility.size(), k, image_site_mobility[k]);
// 			printf("i:%2i s:%2i  dxyz: %g %g %g     %g\n", i,s,dx,dy,dz, image_site_mobility[k]);
// 			printf(">> %g %g  %g  %g\n", dd, m1, normalized_movement, maxMovement);

			const double m2 = sqrt(x*x+y*y+z*z);

			if(m2 > 0)
			{
				x *= m1/m2;
				y *= m1/m2;
				z *= m1/m2;
			}
			
			state_xyz_path[k*3+0] = x;
			state_xyz_path[k*3+1] = y;
			state_xyz_path[k*3+2] = z;
		}
	}
	
	lua_pushnumber(L, absMovement);
	lua_pushnumber(L, maxMovement);
	
	return 2;
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
	const int nn = numberOfImages(); // number of points

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
	const int nn = numberOfImages(); // number of points

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

double MEP::getImageSiteMobility(const int image, const int site)
{
	int k = image * numberOfSites() + site;

	if(k < 0 || k >= image_site_mobility.size())
	{
		printf("bad index in getImageSiteMobility\n");
		return 0;
	}
	
	return image_site_mobility[k];
}


void MEP::setImageSiteMobility(const int image, const int site, double mobility)
{
	int k = image * numberOfSites() + site;
	if(k < 0)
	{
		printf("bad index in setImageSiteMobility\n");
		return;
	}
	while(image_site_mobility.size() <= k)
	{
		image_site_mobility.push_back(1.0);
	}
	
	image_site_mobility[k] = mobility;
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
	lua_call(L, 6, 1);	
	
	if(lua_toboolean(L, -1)) // then a change happened
	{
		energy_ok = false; //need to recalc energy
		good_distances = false;
	}
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
	const int path_length = numberOfImages();

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

double MEP::rightDeriv(vector<double>& src, int i)
{
	double h = 0;
	
	double valHere = src[i];
	
	while(h == 0)
	{
		if(i >= src.size())
			return 0;
		
		h += d12[i];
		i++;
	}
	
	return ( src[i] - valHere ) / h;
}

double MEP::leftDeriv(vector<double>& src, int i)
{
	double h = 0;
	
	double valHere = src[i];
	
	while(h == 0)
	{
		if(i < 0)
			return 0;
		
		h += d12[i-1];
		i--;
	}
	
	return ( valHere - src[i]) / h;
}
	
void MEP::listDerivN(vector<double>& dest, vector<double>& src)
{
	dest.clear();
	if(d12_total == 0)
	{
		for(int i=0; i<src.size(); i++)
			dest.push_back(0);
	}
	else
	{
		dest.push_back( rightDeriv(src, 0) );
		const int num_hyperpoints = d12.size() + 1;
		for(int i=1; i<num_hyperpoints-1; i++)
		{
			const double deLeft  = leftDeriv(src, i);
			const double deRight = rightDeriv(src, i);
			dest.push_back( 0.5 * (deLeft + deRight) );
		}
		dest.push_back( leftDeriv(src, num_hyperpoints-1) );
	}
}


int MEP::calculatePathEnergyNDeriv(lua_State* L, int get_index, int set_index, int energy_index, int n, vector<double>& nderiv)
{
	calculateEnergies(L, get_index, set_index, energy_index); //always returns 0

	double distance = calculateD12(); //total distance
	nderiv.clear();
	
	if(n < 1)
	{
		for(unsigned int i=0; i<energies.size(); i++)
		{
			nderiv.push_back(energies[i]);
		}
		return 0;
	}
	
	if(n == 1)
	{
		listDerivN(nderiv, energies);
		return 0;
	}
	
	// n > 1
	vector<double> nmoDeriv; //n-1 deriv	
	
	calculatePathEnergyNDeriv(L, get_index, set_index, energy_index, n-1, nmoDeriv);

	listDerivN(nderiv, nmoDeriv);
	
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


double MEP::computePointSecondDerivativeAB(lua_State* L, int p, int set_index, int get_index, int energy_index, int c1, int c2)
{
	// back up old sites so we can restore after
	vector<double> cfg;
	saveConfiguration(L, get_index, cfg);
	const int num_sites = numberOfSites();
	double* vxyz = &state_xyz_path[p*num_sites*3];

	const double m1 = sqrt(dot(vxyz,vxyz));
	const double h = 1e-3*m1;
	
// 	printf("(%s:%i) h=%e\n", __FILE__, __LINE__, h);
	
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

#if 1
	const double d1 = (e1-e2) / (2.0*h);
	const double d2 = (e3-e4) / (2.0*h);
	
	const double result = (d1-d2) / (2.0*h);
#else
	const double d1 = (e1-e2);
	const double d2 = (e3-e4);
	
	const double result = (d1-d2) / (4.0*h*h);
#endif
	delete [] state;
	
	loadConfiguration(L, set_index, cfg);	
	
	return result;
}


// partial derivs
void MEP::computePointSecondDerivative(lua_State* L, int p, int set_index, int get_index, int energy_index, double* derivsAB)
{
	const int num_sites = numberOfSites();
	int deriv_pos = 0;
	for(int c1=0; c1<num_sites*3; c1++)
	{
		for(int c2=0; c2<num_sites*3; c2++)
		{
			derivsAB[deriv_pos] = computePointSecondDerivativeAB(L, p, set_index, get_index, energy_index, c1, c2);
			deriv_pos++;
		}
	}
}

	
// relax individual point. This is used to refine a maximal point
// expected on the stack:
// at 1, mep
// at 2, point number
// at 3, function get_site_ss1(x,y,z) return sx,sy,sz 
// at 4, function set_site_ss1(x,y,z, sx,sy,sz)  return something_changed
// at 5, function get_energy_ss1()
// at 6, number of steps
// at 7, optional direction to negate as a table
// returns energy and energy change due to relax
int MEP::relaxSinglePoint(lua_State* L)
{
	const int num_sites = numberOfSites();
	const int path_length = numberOfImages();

	const int p = lua_tointeger(L, 2) - 1;
	if(p < 0 || p >= path_length)
		return luaL_error(L, "Point index out of bounds");

	const int get_index = 3;
	const int set_index = 4;
	const int energy_index = 5;

	
	if(!lua_isfunction(L, 3) || !lua_isfunction(L, 4) || !lua_isfunction(L, 5))
		return luaL_error(L, "3 functions expected - perhaps you should call the wrapper method :relaxSinglePoint");
	
	int num_steps = lua_tointeger(L, 6);
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
	if(lua_istable(L, 7))
	{
		neg_dir = new double[3 * num_sites];
		for(int i=0; i<3 * num_sites; i++)
		{
			lua_pushinteger(L, i+1);
			lua_gettable(L, 7);
			neg_dir[i] = lua_tonumber(L, -1);
			lua_pop(L, 1);
		}
	}

	double* grad = new double[3 * num_sites];
	double maxMovement = 0;
	double absMovement = 0;
		
	// here's the loop over num_steps
	for(int i=0; i<num_steps; i++)
	{
		// first compute e. grad around point
		
		for(int s=0; s<num_sites; s++)
		{
			computePointGradAtSite(L, p, s, set_index, energy_index, &grad[s*3]);
		}

		if(neg_dir)
		{
			const double n2 = dot(neg_dir, neg_dir, 3*num_sites);
			const double proj = dot(neg_dir, grad, 3*num_sites);
			
			if(n2 != 0)
			{
				for(int j=0; j<3*num_sites; j++)
				{
					neg_dir[j] *= proj/n2;
					grad[j] -= 2.0*neg_dir[j];
				}
			}

		}
		
		double vv[3];
		double nv[3];
		double delta[3];
		double pdelta[3];
		for(int s=0; s<num_sites*3; s+=3)
		{
			const double m1 = sqrt(dot(&vxyz[s],&vxyz[s],3));

			for(int j=0; j<3; j++)
			{
				delta[j] = grad[s+j] * beta;
				vv[j] = vxyz[s+j] + delta[j];
			}

			normalizeTo(&vxyz[s+0], vv, m1);

			const double dd = sqrt(dot(delta, delta));
			const double normalized_movement = dd/m1;

			absMovement += normalized_movement;
			if(maxMovement < normalized_movement)
				maxMovement = normalized_movement;
		}
	}
	
	delete [] grad;
	
	if(neg_dir)
		delete [] neg_dir;
	
	// write updated cfg
 	setAllSpins(L, set_index, vxyz);
	
// 	double final_energy = getEnergy(L, energy_index);
// 
// 	if(energies.size() < path_length)
// 	{
// 		energies.clear();
// 		for(int p=0; p<path_length; p++)
// 		{
// 			double e = 0;
// 			for(int s=0; s<num_sites; s++)
// 			{
// 				setSiteSpin(L, set_index, &sites[s*3], &state_xyz_path[p * sites.size() + s*3]);
// 			}
// 
// 			energies.push_back(getEnergy(L, energy_index));
// 		}	
// 	}
// 	else
// 	{
// 		energies[p] = final_energy;
// 	}
	
	// need to restore saved configuration to SpinSystem
	loadConfiguration(L, set_index, cfg);

// 	lua_pushnumber(L, initial_energy);
// 	lua_pushnumber(L, final_energy);
// 	lua_pushnumber(L, fabs(initial_energy - final_energy));
	
	lua_pushnumber(L, absMovement);
	lua_pushnumber(L, maxMovement);
	
// 	energy_ok = false; //gets updates in setAllSpins->setSiteSpin
// 	good_distances = false;
	return 2;
}

int MEP::numberOfImages()
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
		
// 	printf("(%s:%i) %e %e\n", __FILE__, __LINE__, vxyz[0], grad3[0]);
		
	grad3[0] = (g[1] - g[0]) / (2.0 * epsilon);
	grad3[1] = (g[3] - g[2]) / (2.0 * epsilon);
	grad3[2] = (g[5] - g[4]) / (2.0 * epsilon);
	
// 	printf("%e %e %e --> %e %e %e\n", vxyz[0], vxyz[1], vxyz[2], grad3[0], grad3[1], grad3[2]);
}

void MEP::writePathPoint(lua_State* L, int set_index, double* vxyz)
{
	const int num_sites = numberOfSites();
	
	// write path point as it is given
	for(int s=0; s<num_sites; s++)
	{
		setSiteSpin(L, set_index, &sites[s*3], &vxyz[s*3]);
	}
	
// 	energy_ok = false;
// 	good_distances = false;
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
	const int path_length = numberOfImages();
	
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
	const int ns = numberOfSites();
	const int ni = numberOfImages();
	
	force_vector.resize( state_xyz_path.size() ); //since we're doing random-ish access
	
	// need to save current configuration
	vector<double> cfg;
	saveConfiguration(L, get_index, cfg);
	
	// lets march along the path
	for(int i=0; i<ni; i++)
	{
		double* vxyz = &state_xyz_path[i*ns*3];
		
		// write path point as it currently stands
		writePathPoint(L, set_index, &state_xyz_path[i*ns*3]);

		for(int s=0; s<ns; s++)
		{
			double grad3[3];
			computePointGradAtSite(L, i, s, set_index, energy_index, grad3);

			force_vector[i*ns*3 + s*3 + 0] = beta * grad3[0];
			force_vector[i*ns*3 + s*3 + 1] = beta * grad3[1];
			force_vector[i*ns*3 + s*3 + 2] = beta * grad3[2];
			/*
			printf("i:%2i s:%2i  fv: %g %g %g\n", i,s,
				   force_vector[i*ns*3 + s*3 + 0],
					   force_vector[i*ns*3 + s*3 + 1],
						   force_vector[i*ns*3 + s*3 + 2]);
						   */
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
	return mep->applyForces(L);	
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

static int l_setimagesitemobility(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	
	const int i = lua_tointeger(L, 2)-1;
	const int s = lua_tointeger(L, 3)-1;
	const double m = lua_tonumber(L, 4);
	
	mep->setImageSiteMobility(i, s, m);
	
	return 0;
}


static int l_resampleStateXYZPath(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	return mep->resampleStateXYZPath(L);
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
// at 6, number of steps
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
// this is the internal version
static int l_computepoint2deriv(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);

	const int get_index = 3;
	const int set_index = 4;
	const int energy_index = 5;
	
	int num_sites = mep->numberOfSites();
	
	int p = lua_tointeger(L, 2) - 1;
	
	if(p < 0 || p >= mep->numberOfImages())
		return luaL_error(L, "Invalid point number");

	double* derivsAB = new double[num_sites * num_sites * 9];
	
	mep->computePointSecondDerivative(L, p, set_index, get_index, energy_index, derivsAB);
	
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
		return luaL_error(L, "Path Point or site is out of bounds. {Point,Site} = {%d,%d}. Upper Bound = {%d,%d}", p+1,s+1,mep->numberOfImages(), mep->numberOfSites());
	}

	int idx = mep->sites.size() * p + s*3;

	if(mep->state_xyz_path.size() < idx+2)
	{
		return luaL_error(L, "Path Point or site is out of bounds. {Point,Site} = {%d,%d}. Upper Bound = {%d,%d}", p+1,s+1,mep->numberOfImages(), mep->numberOfSites());
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

	
static int l_setsite(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);

	const int p = lua_tointeger(L, 2) - 1;
	const int s = lua_tointeger(L, 3) - 1;

	if(p < 0 || s < 0)
	{
		return luaL_error(L, "Path Point or site is out of bounds. {Point,Site} = {%d,%d}. Upper Bound = {%d,%d}", p+1,s+1,mep->numberOfImages(), mep->numberOfSites());
	}

	int idx = mep->sites.size() * p + s*3;

	if(mep->state_xyz_path.size() < idx+2)
	{
		return luaL_error(L, "Path Point or site is out of bounds. {Point,Site} = {%d,%d}. Upper Bound = {%d,%d}", p+1,s+1,mep->numberOfImages(), mep->numberOfSites());
	}

	double x,y,z;
	
	if(lua_istable(L, 4))
	{
		lua_pushinteger(L, 1);
		lua_gettable(L, 4);
		x = lua_tonumber(L, -1);
		lua_pop(L, 1);

		lua_pushinteger(L, 2);
		lua_gettable(L, 4);
		y = lua_tonumber(L, -1);
		lua_pop(L, 1);

		lua_pushinteger(L, 3);
		lua_gettable(L, 4);
		z = lua_tonumber(L, -1);
		lua_pop(L, 1);
	}
	else
	{
		x = lua_tonumber(L, 4);
		y = lua_tonumber(L, 5);
		z = lua_tonumber(L, 6);
	}
	
	mep->state_xyz_path[idx+0] = x;
	mep->state_xyz_path[idx+1] = y;
	mep->state_xyz_path[idx+2] = z;

	mep->energy_ok = false;
	mep->good_distances = false;
	
	return 0;
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

static int l_setbeta(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
	mep->beta = lua_tonumber(L, 2);
	return 0;
}

static int l_getbeta(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
	lua_pushnumber(L, mep->beta);
	return 1;
}

static int l_internal_copyto(lua_State* L)
{
    LUA_PREAMBLE(MEP, src, 1);
    LUA_PREAMBLE(MEP, dest, 2);

	src->internal_copy_to(dest);
	return 0;
}

static int l_maxpoints(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	return mep->maxpoints(L);	
}

static int l_absoluteDifference(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep1, 1);
	LUA_PREAMBLE(MEP, mep2, 2);

	int point = -1;
	if(lua_isnumber(L, 3))
	{
		point = lua_tointeger(L, 3) - 1; // c->lua base change
	}

	double max_diff;
	const double d = mep1->absoluteDifference(mep2, point, max_diff);

	if(d == -1)
		return luaL_error(L, "internal state size mismatch");

	lua_pushnumber(L, d);
	lua_pushnumber(L, max_diff);
	return 2;
}

static int l_debug(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);

	lua_pushinteger(L, mep->state_xyz_path.size());
	return 1;
}

static int l_ppc(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);

	lua_pushinteger(L, mep->numberOfImages());
	return 1;
}

static int l_pend_(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	
	int n = 0;
	if(lua_isnumber(L, 2))
		n = lua_tonumber(L, 2);

	const int get_index = 3;
	const int set_index = 4;
	const int energy_index = 5;
	
	vector<double> nderiv;

	mep->calculatePathEnergyNDeriv(L, get_index, set_index, energy_index, n, nderiv);
	
	double total = mep->d12_total;
	
	lua_newtable(L); //energy
	lua_newtable(L); //location
	

	vector<double> normalized_distance;
	normalized_distance.push_back(0);
	for(int i=0; i<mep->d12.size(); i++)
	{
		normalized_distance.push_back(normalized_distance[i] + mep->d12[i] / mep->d12_total);
	}
	
	for(int i=0; i<nderiv.size(); i++)
	{
		const double ederiv_here = nderiv[i];
		const double pos_here = normalized_distance[i];
		
		lua_pushinteger(L, i+1);
		lua_pushnumber(L, ederiv_here);
		lua_settable(L, -4);
		
		lua_pushinteger(L, i+1);
		lua_pushnumber(L, pos_here);
		lua_settable(L, -3);
	}
	
	lua_newtable(L);
	int j = 0;
	for(int i=1; i<nderiv.size(); i++)
	{
		if( (nderiv[i] * nderiv[i-1]) < 0) //sign change
		{
			const double v0 = nderiv[i-1];
			const double v1 = nderiv[i];
			
			const double p0 = normalized_distance[i-1];
			const double p1 = normalized_distance[i];
			
			const double t = -v0 * ((p1-p0) / (v1-v0));
			
			const double pCrossing = p0 + t * (p1-p0);
			
			j++;
			lua_pushinteger(L, j);
			lua_pushnumber(L, pCrossing);
			lua_settable(L, -3);
		}
	}

	return 3;
}


int MEP::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Calculates a minimum energy pathway between two states.");
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
	if(func == l_setsite)
	{
		lua_pushstring(L, "Set site x,y and z coordinates.");
		lua_pushstring(L, "2 Integers, 3 Numbers or 1 table of 3 Numbers: 1st integer is path index, 2nd integer is site index, 3 numbers are the new x,y and z coordinates.");
		lua_pushstring(L, "");
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
	if(func == l_ppc)
	{
		lua_pushstring(L, "Get number of path points that currently exist in the object.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: number of path points.");
		return 3;
	}
	
	if(func == l_absoluteDifference)
	{
		lua_pushstring(L, "Get the absolute difference between the calling MEP and the passed MEP. Difference between two moments m1 and m2 is defined as the norm of the vector D where D = normalized(m1) - normalized(m2).");
		lua_pushstring(L, "1 MEP: MEP to compare against.");
		lua_pushstring(L, "2 Numbers: Total of all differences and the maximum single difference.");
		return 3;
	}

	if(func == l_setbeta)
	{
		lua_pushstring(L, "Set the internal scaling factor that converts between energy gradient and displacement. This value gets adapted automatically. If a close value for Beta is knows at the start of the calculation it will save a few iterations if it is set explicitly.");
		lua_pushstring(L, "1 Number: Beta");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_getbeta)
	{
		lua_pushstring(L, "Get the internal scaling factor that converts between energy gradient and displacement. This value gets adapted automatically.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: Beta");
		return 3;
	}
// 	if(func == l_pend)
// 	{
// 		lua_pushstring(L, "Calculate the Nth derivative of the path energy.");
// 		lua_pushstring(L, "1 Integer: The derivative, 0 = energy, 1 = slope of energy, 2 = acceleration of energy, ...");
// 		lua_pushstring(L, "2 Tables: The Nth derivatives of the energy, the normalized location along the path between 0 and 1.");
// 		return 3;
// 	}
	
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
		{"pathPointCount", l_ppc},
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
		{"setSpin", l_setsite},
		{"applyForces", l_applyforces},
		{"_addStateXYZ", l_addstatexyz},
		{"_setImageSiteMobility", l_setimagesitemobility},
		{"_relaxSinglePoint", l_relaxSinglePoint_}, //internal method
		{"_hessianAtPoint", l_computepoint2deriv},
		{"_maximalPoints", l_maxpoints},
		{"_randomize", l_randomize_},
		{"internalCopyTo", l_internal_copyto},
		{"absoluteDifference", l_absoluteDifference},
		{"setBeta", l_setbeta},
		{"beta", l_getbeta},
		{"_debug", l_debug},
		{"_pathEnergyNDeriv", l_pend_},
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
