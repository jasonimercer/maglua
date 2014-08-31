#include "mep_bestpath.h"
#include "vec_cs.h"
#include <vector>
#include <list>
#include <algorithm>    // std::sort
#include <math.h>

#define report 0
//#define report stdout

#ifndef M_PI
#define M_PI 3.14159265359
#endif

using namespace std;

class site
{
public:
    site() {source=-1;considered=false;}
    site(const site& s) {set(s.vs);source=-1;considered=false;}
    site(const vector<VectorCS>& s){set(s);source=-1;considered=false;}
    void set(const vector<VectorCS>& s) {vs.clear(); for(int i=0; i<s.size(); i++) vs.push_back(VectorCS(s[i]));}
    vector <VectorCS> vs;

    bool considered; // used in the path search
    int source; // point backward to rebuild path;
    double energy;
    vector<int> neighbour;
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

static double angle_between_sites(const site& a, const site& b)
{
#if 0
    double maxa = VectorCS::angleBetween(a.vs[0], b.vs[0]);
    for(int q=1; q<a.vs.size(); q++)
    {
        const double aa = VectorCS::angleBetween(a.vs[q], b.vs[q]);
        if(aa > maxa)
            maxa = aa;
    }
    return maxa;
#else
    double sum = 0;
    for(int q=0; q<a.vs.size(); q++)
    {
	sum += VectorCS::angleBetween(a.vs[q], b.vs[q]);
    }
    return sum;
#endif
}

static double angle_between_sites(const vector<site>& sites, int i, int j)
{
    return angle_between_sites(sites[i], sites[j]);
}

typedef std::pair < double, int > doubleIntPair;

static bool doubleIntPairOrder(const doubleIntPair& i,const doubleIntPair& j) 
{
    return i.first < j.first;
}



// find the smallest energy increment from the border
static int fill_up(list<int>& border, vector<site>& sites)
{
    list<int>::iterator it;

    int low_e_b = -1; // border index
    int low_e_n = -1; // neighbour index

    for(it=border.begin(); it!=border.end(); it++)
    {
        for(int i=0; i<sites[*it].neighbour.size(); i++)
        {
            int n = sites[*it].neighbour[i];
            if(sites[n].considered == false)
            {
                if(low_e_n == -1)
                {
                    low_e_b = *it;
                    low_e_n = n;
                }
                else
                {
                    if(sites[n].energy < sites[low_e_n].energy)
                    {
                        low_e_b = *it;
                        low_e_n = n;
                    }
                }
            }
        }
    }

    if(low_e_b != -1)
    {
        if(report)
            fprintf(report, "(%s:%i) Up:    Adding %d to border from %d\n", __FILE__,__LINE__,low_e_n,low_e_b);
                    
        border.push_back(low_e_n);
        sites[low_e_n].considered = true;
        sites[low_e_n].source = low_e_b;
        
        return low_e_n;
    }
    return -1;
}

// find the lowest energy border with a low energy unconsidered neighbour
static int fill_down(list<int>& border, vector<site>& sites)
{
    list<int>::iterator it;

    int low_e_b = -1; // border index
    int low_e_n = -1; // neighbour index

    for(it=border.begin(); it!=border.end(); it++)
    {
        if(low_e_b == -1 || (sites[*it].energy < sites[low_e_b].energy))
        {
            // looking for lowest energy unconsidered neighbour
            int low_un = -1;
            for(int i=0; i<sites[*it].neighbour.size(); i++)
            {
                int n = sites[*it].neighbour[i];
                if(sites[n].considered == false)
                {
                    if(low_un == -1 || sites[n].energy < sites[low_un].energy)
                    {
                        low_un = n;
                    }
                }
            }

            if(low_un != -1)
            {
                // if this site has a low energy unconsidered neighbour
                if( sites[low_un].energy < sites[*it].energy )
                {
                    if(low_e_b == -1)
                    {
                        low_e_b = *it;
                        low_e_n = low_un;
                    }
                    else
                    {
                        if( sites[*it].energy < sites[low_e_b].energy )
                        {
                            low_e_b = *it;
                            low_e_n = low_un;
                        }
                    }
                }
            }
        }
    }

    if(low_e_b != -1)
    {
        if(report)
            fprintf(report, "(%s:%i) Down:  Adding %d to border from %d\n", __FILE__,__LINE__,low_e_n,low_e_b);
                    
        border.push_back(low_e_n);
        sites[low_e_n].considered = true;
        sites[low_e_n].source = low_e_b;
        
        return low_e_n;
    }
    return -1;
}

static int inside_border(int idx, vector<site>& sites)
{
    int all_considered = sites[idx].considered;

    for(int i=0; i<sites[idx].neighbour.size() && all_considered; i++)
    {
        int k = sites[idx].neighbour[i];

        if(sites[k].considered == false)
            all_considered = 0;
    }
    return all_considered;
}

static void reduce_border(list<int>& border, vector<site>& sites)
{
    for(list<int>::iterator it = border.begin(); it != border.end(); ++it)
    {
        if(inside_border(*it, sites))
        {
            border.remove(*it);
            return reduce_border(border, sites);
        }
    }
}

int l_bestpath(lua_State* L, int base)
{
    const int ors = 2; // ors_combinations table (orientation combinations)
    const int eni = 3; // energy table index
    const int sidx = 4; // start index
    const int eidx = 5; // end index
    const int con_deg = 6; // goal connectivity degree index

    vector<site> sites;
    vector<double> energies;
    site start;
    site end;
    int conDegree = lua_tointeger(L, con_deg);

    lua_pushnil(L);
    while(lua_next(L, ors)) // for each site
    {
	sites.push_back(site());
	site& s = sites.back();
        s.considered = false; // used in path search
	s.source = -1; // used backwards to rebuild path

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


    if(sites.size() != energies.size())
    {
        return luaL_error(L, "Sites and Energies lists do not have the same size");
    }

    for(int i=0; i<sites.size(); i++)
    {
        sites[i].energy = energies[i];
    }


    get_site(L, sidx, start);
    get_site(L, eidx, end);

    // need to look through the orientations for the 
    // sites closest to the start and end
    int closest_start_idx = -1;
    int closest_end_idx = -1;

    double closest_start_dist = 1e10;
    double closest_end_dist = 1e10;

    for(int i=0; i<sites.size(); i++)
    {
        double a1 = angle_between_sites(start, sites[i]);
        double a2 = angle_between_sites(  end, sites[i]);

        if(closest_start_idx == -1 || a1 < closest_start_dist)
        {
            closest_start_dist = a1;
            closest_start_idx = i;
        }

        if(closest_end_idx == -1 || a2 < closest_end_dist)
        {
            closest_end_dist = a2;
            closest_end_idx = i;
        }
    }


    // for each site we want the "conDegree" closest sites
    for(int i=0; i<sites.size(); i++)
    {
        vector< doubleIntPair > angleIndex;

        for(int j=0; j<sites.size(); j++)
        { 
            if(i != j)
            {
                angleIndex.push_back( doubleIntPair(angle_between_sites(sites, i, j), j) );
                
                std::sort (angleIndex.begin(), angleIndex.end(), doubleIntPairOrder);
                
                while(angleIndex.size() > conDegree)
                {
                    angleIndex.pop_back();
                }
            }
        }

        for(int j=0; j<angleIndex.size(); j++)
        {
            sites[i].neighbour.push_back(angleIndex[j].second);
        }
    }


    if(report)
    {
        for(int c=0; c<sites.size(); c++)
        {
            printf("%3d > ", c);
            for(int q=0; q<sites[c].neighbour.size(); q++)
            {
                printf("%3d (%5g)", sites[c].neighbour[q], angle_between_sites(sites[c], sites[ sites[c].neighbour[q] ]));
            }
            printf("\n");
        }
    }

    list<int> border;
    border.push_back(closest_start_idx);

    sites[closest_start_idx].considered = true;
    sites[closest_start_idx].source = closest_start_idx;

    if(report)
        fprintf(report, "(%s:%i) Prime: Adding %d to border from %d\n", __FILE__,__LINE__,closest_start_idx,closest_start_idx);

    int add_fill_up   = 1;
    int add_fill_down = 1;
    if(report)
        fprintf(report, "(%s:%i) %i %i %i\n", __FILE__, __LINE__, (add_fill_up != -1), (add_fill_down != -1), (sites[closest_end_idx].source));



    int progress = 1;
    while(progress && (sites[closest_end_idx].source == -1))
    {
        add_fill_up   = 1;
        add_fill_down = 1;
        progress = 0;

        while(add_fill_down != -1)
        {
            add_fill_down = fill_down(border, sites);
            if(add_fill_down != -1)
                progress = 1;
        }

        add_fill_up = fill_up(border,sites);

        reduce_border(border,sites);

        if(add_fill_up != -1)
            progress = 1;

        if(report)
            fprintf(report, "(%s:%i) %i %i\n", __FILE__, __LINE__, progress, sites[closest_end_idx].source);
    }

    

    if(sites[closest_end_idx].source == -1)
    {
        lua_pushboolean(L, false);
        return 1;
    }

    {
        list<int> v;
        int i = closest_end_idx;

        while(i != closest_start_idx)
        {
            // printf("i = %i\n", i);
            v.push_front(i);
            i = sites[i].source;
        }
        v.push_front(i);
 
        
        lua_newtable(L);
        int k = 1;
        for(list<int>::iterator it = v.begin(); it != v.end(); it++)
        {
            lua_pushinteger(L, k);
            lua_pushinteger(L, (*it)+1);
            lua_settable(L, -3);
            k++;
        }
        return 1;
    }
}

