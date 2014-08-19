#include "mep_bestpath.h"
#include "vec_cs.h"
#include <vector>
#include <map>
#include <deque>
#include <algorithm>    // std::sort
#include <math.h>


#ifndef M_PI
#define M_PI 3.14159265359
#endif

using namespace std;

class site
{
public:
    site() {}
    site(const site& s) {set(s.vs);}
    site(const vector<VectorCS>& s){set(s);}
    void set(const vector<VectorCS>& s) {vs.clear(); for(int i=0; i<s.size(); i++) vs.push_back(VectorCS(s[i]));}
    vector <VectorCS> vs;
};

static int get_site(lua_State* L, int idx, site& s)
{
    lua_pushnil(L);
    while(lua_next(L, idx)) // for each orientation in site
    {
	double x[3];
	for(int i=0; i<3; i++)
	{
	    lua_pushinteger(L, i+1);
	    lua_gettable(L, -2);
	    x[i] = lua_tonumber(L, -1);
	    lua_pop(L, 1);
	}
	s.vs.push_back(VectorCS(x, Cartesian));
	lua_pop(L, 1);
    }
}

double angle_between_sites(const site& a, const site& b)
{
    double sum = 0;
    for(int q=0; q<a.vs.size(); q++)
    {
	sum += VectorCS::angleBetween(a.vs[q], b.vs[q]);
    }
    return sum;
}

double angle_between_sites(const vector<site>& sites, int i, int j)
{
    return angle_between_sites(sites[i], sites[j]);
}


int l_bestpath(lua_State* L, int base)
{
    const int orc = 2; // ors_combinations table
    const int eni = 3; // energy table index
    const int sidx = 4; // start index
    const int eidx = 5; // end index
    vector<site> sites;
    vector<double> energies;
    site start;
    site end;

    lua_pushnil(L);
    while(lua_next(L, orc)) // for each site
    {
	sites.push_back(site());
	site& s = sites.back();
	
	int idx = lua_gettop(L);
	get_site(L, idx, s);
	lua_pop(L, 1);
    }
    int nsites = sites.size();
    int ops = sites[0].vs.size(); // orientations per site

    lua_pushnil(L);
    while(lua_next(L, eni)) // for each energy
    {
	energies.push_back(lua_tonumber(L, -1));
	lua_pop(L, 1);
    }

    get_site(L, sidx, start);
    get_site(L, eidx, end);

    // with all the orientations, we need to figure out which are neighbours. 
    // we'll make a histogram of total angle (degrees) between x 100 as an int
    std::map<int, std::size_t> histogram;
    const double conv = 180.0/M_PI * 100;
    for(int i=0; i<sites.size(); i++)
    {
	for(int j=0; j<sites.size(); j++)
	{
	    double a = angle_between_sites(sites, i, j) * conv;
	    histogram[a]++;
	}
    }

    // looking for sites closest to start and end
    double closest_idx = 0;
    double closest_angle = angle_between_sites(sites[0], start);
    for(int i=1; i<sites.size(); i++)
    {
	double a = angle_between_sites(sites[i], start);
	if(a < closest_angle)
	{
	    closest_angle = a;
	    closest_idx = i;
	}
    }

    int start_idx = closest_idx;

    closest_idx = 0;
    closest_angle = angle_between_sites(sites[0], end);
    for(int i=1; i<sites.size(); i++)
    {
	double a = angle_between_sites(sites[i], end);
	if(a < closest_angle)
	{
	    closest_angle = a;
	    closest_idx = i;
	}
    }

    int end_idx = closest_idx;


    // not sure that iterating through the histogram will step from smallest to largest key
    // making a vector and sorting it
    typedef std::map<int, std::size_t>::iterator it_type;
    vector<int> keys;
    for(it_type iterator = histogram.begin(); iterator != histogram.end(); iterator++) 
    {
	keys.push_back(iterator->first);
    }

    sort(keys.begin(), keys.end());

    // would like to have each site be connected to 3 * orientations per site sites
    int total = -nsites; // excluding self connection
    int cuttoff = -1;
    int goal = nsites * ops * 3;
    for(int i=0; i<keys.size() && cuttoff<0; i++)
    {
	total = total + histogram[ keys[i] ];
	if(total >= goal)
	{
	    cuttoff = keys[i];
	}
    }
    
    vector< vector<int> > connectivity;

    for(int i=0; i<sites.size(); i++)
    {
	connectivity.push_back(vector<int>());
    }

    for(int i=0; i<sites.size()-1; i++)
    {
        for(int j=i+1; j<sites.size(); j++)
        {
	    int a = (int) angle_between_sites(sites, i, j) * conv;
	    
	    if(a <= cuttoff)
	    {
		connectivity[i].push_back(j);
		connectivity[j].push_back(i);
	    }
        }
    }

    for(int c=0; c<connectivity.size(); c++)
    {
	printf("%3d > ", c);
	for(int q=0; q<connectivity[c].size(); q++)
	{
	    printf("%3d ", connectivity[c][q]);
	}
	printf("\n");
    }

    // initialize graph: invalid sources. 
    typedef std::pair<int, double> src_energy;
    vector< src_energy > graph;
    for(int i=0; i<sites.size(); i++)
    {
	graph.push_back(src_energy(-1,0));
    }


    // flood-fill from start_idx until hit end_idx

    deque<int> border;
    border.push_back(start_idx);

    while(1)
    {
	// look at all un-traversed neighbours of the border. Find smallest energy
	
    }



#if 0
    for(it_type iterator = histogram.begin(); iterator != histogram.end(); iterator++) 
    {
	// printf("%5d %5d\n", iterator->second, iterator->first);
	printf("%5d %5d\n", iterator->first, iterator->second);
    }
#endif
    return 0;
}

