#include "mep_bestpath.h"
#include "vec_cs.h"
#include "mep_spheres.h"
#include "mep.h"

#include <map>
#include <string.h>
#include <vector>
#include <list>
#include <algorithm>    // std::sort
#include <math.h>


typedef vector<int> vint;

static double findPoint(const sphere* sph, VectorCS& p, int& pidx)
{
    double min_angle = VectorCS::angleBetween(p, VectorCS(sph[0].vertex));
    pidx = 0;

    int i=1;
    while(sph[i].neighbours)
    {
        double a = VectorCS::angleBetween(p, VectorCS(sph[i].vertex));
        if(a < min_angle)
        {
            min_angle = a;
            pidx = i;
        }
        i++;
    }

    return min_angle;
}

static double findHyperPoint(vector <const sphere*>& spheres, vector<VectorCS>& p, vint& pidx)
{
    double sum_min_angle = 0;

    for(int i=0; i<p.size(); i++)
    {
        pidx.push_back(0);
        sum_min_angle += findPoint(spheres[i], p[i], pidx[i]);
    }
    return sum_min_angle;
}

static bool vint_same(const vint& a, const vint& b)
{
    if(a.size() != b.size())
        return false;

    for(int i=0; i<a.size(); i++)
    {
        if(a[i] != b[i])
            return false;
    }
    return true;
}

static bool v_inc(vector<int>& state, vector<int>& max, int pos=0)
{
    if(pos >= state.size())
        return false;
    state[pos]++;
    if(state[pos] == max[pos])
    {
        state[pos] = 0;
        return v_inc(state, max, pos+1);
    }
    return true;
}

static void pick_combination(const vector<int>& state, vector<const int*>& srcs, vint& dst)
{
    dst.clear();

    for(int i=0; i<state.size(); i++)
    {
        dst.push_back( srcs[i][state[i]] );
    }
}

static void all_combinations(vector<vint>& dest, vector<const int*>& srcs)
{
    vector<int> state;
    vector<int> maxs;
    for(int i=0; i<srcs.size(); i++)
    {
        state.push_back(0);
        
        int max = -1;
        while(srcs[i][max+1] != -1)
            max++;
        maxs.push_back(max);
    }
    
    dest.push_back(vint());

    pick_combination(state, srcs, dest.back());

    while(v_inc(state, maxs))
    {
        dest.push_back(vint());
        pick_combination(state, srcs, dest.back());
    }


}

class dijkstrasNode
{
public:
    dijkstrasNode() 
        {
            source_energy = INFINITY; 
            energy_calculated=false;
            considered = false;
        }
    dijkstrasNode(const dijkstrasNode& n)
        {
            energy_calculated = n.energy_calculated;
            energy = n.energy;
            source_energy = n.source_energy;
            source = n.source;
            considered = n.considered;
            neighbour_list = n.neighbour_list;
        }
    double getEnergy(MEP* mep, const vint& idxs, vector<const sphere*>& spheres)
        {
            if(energy_calculated)
                return energy;

            vector<VectorCS> cfg;

            for(int i=0; i<idxs.size(); i++)
                cfg.push_back(VectorCS(spheres[i][idxs[i]].vertex));

            energy = mep->energyOfCustomPoint(cfg);
            energy_calculated = true;
            return energy;
        }
    
    void create_neighbour_list(const vint& here, vector<const sphere*>& spheres)
        {
            if(neighbour_list.size() > 0)
                return;

            int n = here.size();

            // static void all_combinations(vector<vint>& dest, vector<int*>& srcs)
           
            vector<const int*> srcs;
            for(int i=0; i<n; i++)
            {
                int j = here[i];
                srcs.push_back(spheres[i][j].neighbours);
            }

            all_combinations(neighbour_list, srcs);

        }
    

    bool energy_calculated;
    double energy;
    double source_energy;
    vint source;

    vector<vint> neighbour_list;

    bool considered;

};

class dijkstrasSearch
{
public:
    dijkstrasSearch() {step = 0;};

    vint getSource(vint& n)
        {
            return nodes[n].source;
        }

    bool iterate()
        {
            if(edge.empty())
            {
                //printf("empty\n");
                return false;
            }

            vint v = edge.front();
            edge.pop_front();


            if( nodes.find(v) == nodes.end() ) // then doesn't contain
            {
                nodes.insert ( std::pair<vint,dijkstrasNode>(v,dijkstrasNode()) );
            }

            dijkstrasNode& node = nodes.find(v)->second;

            if(node.considered)
            {
                //printf("considered\n");
                return true;
            }

            node.create_neighbour_list(v, spheres);

            if(step == 0)
            {
                node.source_energy = 0;
            }
            step++;


            double energy_here = node.getEnergy(mep, v, spheres);

            for(int i=0; i<node.neighbour_list.size(); i++)
            {
                vint& nn = node.neighbour_list[i];

                if( nodes.find(nn) == nodes.end() ) // then doesn't contain
                {
                    nodes.insert ( std::pair<vint,dijkstrasNode>(nn,dijkstrasNode()) );
                }

                dijkstrasNode& neighbour = nodes[nn];

                if(!neighbour.considered)
                {
                    double e = energy_here;
                    if(node.source_energy > e)
                        e = node.source_energy;
                    if(e < neighbour.source_energy)
                    {
                        neighbour.source_energy = e * 1.0000000001; // this will select for shortest paths down-hill
                        neighbour.source = v;
                    }
                    edge.push_back(nn);

                    /*
                    for(int q=0; q<nn.size(); q++)
                    {
                        printf("%4d ", nn[q]);
                    }
                    printf("\n");
                    */
                }
            }
            node.considered = true;
            return true;
        };

    
    std::map<vint, dijkstrasNode> nodes;
    std::list<vint> edge;

    vector<const sphere*> spheres;    
    MEP* mep;
    int step;
};

static void sphere_lookup(vint& v, vector<const sphere*>& spheres, vector<double>& mags, vector<VectorCS>& cfg)
{
    cfg.clear();

    for(int i=0; i<v.size(); i++)
    {
        VectorCS vcs(spheres[i][v[i]].vertex);
        // printf("mags[%i] = %g\n", i, mags[i]);
        vcs.setMagnitude(mags[i]);
        cfg.push_back(vcs);//.copy());
    }
}

int l_bestpath(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    int n = 1; // sphere level
    int ns = mep->numberOfSites();

    if(!lua_istable(L, 2) || !lua_istable(L, 3))
    {
        return luaL_error(L, "Dijkstra's path needs start and end.");
    }

    vector<VectorCS> a;
    vector<VectorCS> b;

    lua_pushnil(L);
    while(lua_next(L, 2))
    {
        a.push_back(lua_toVectorCS(L, lua_gettop(L)));
        //print_vec(a.back());
        lua_pop(L, 1);
    }

    lua_pushnil(L);
    while(lua_next(L, 3))
    {
        b.push_back(lua_toVectorCS(L, lua_gettop(L)));
        //print_vec(b.back());
        lua_pop(L, 1);
    }

    vector<double> mags;

    for(int i=0; i<a.size(); i++)
        mags.push_back(a[i].magnitude());


    if(lua_isnumber(L, 4))
        n = lua_tointeger(L, 4);

    if(n < 1 || n > 5)
        return luaL_error(L, "Dodecahedron subdivision level must be between 1 and 5.");

    vector<const sphere*> spheres;    
    for(int i=0; i<ns; i++)
        spheres.push_back(get_sphere(n));

    
    vint a_idx;
    vint b_idx;

    double sum_a = findHyperPoint(spheres, a, a_idx);
    double sum_b = findHyperPoint(spheres, b, b_idx);


    dijkstrasSearch ds;

    ds.mep = mep;
    ds.spheres = spheres;
    ds.edge.push_back(a_idx);

    //int dsi = 0;
    while(ds.iterate())
    {
        //printf("dsi %d\n", dsi);
        //dsi++;
    }

    list<vint> soln;

    soln.push_front(b_idx);
    
    int path = 0;
    while(!vint_same(b_idx, a_idx))
    {
        //printf("path %d\n", path);
        //path++;
        b_idx = ds.getSource(b_idx);
        soln.push_front(b_idx);
    }

    lua_newtable(L);

    int i=1;
    
    vector<VectorCS> cfg;
    list<vint>::iterator it;
    for(it=soln.begin(); it!=soln.end(); ++it)
    {
        vint& v = *it;


        lua_pushinteger(L, i);

        lua_newtable(L);
        sphere_lookup(v, spheres, mags, cfg);

        for(int j=0; j<cfg.size(); j++)
        {
            lua_pushinteger(L, j+1);
            
            lua_pushVectorCS(L, cfg[j], VCSF_ASTABLE | VCSF_CSDESC);
            lua_settable(L, -3);
        }
        lua_settable(L, -3);
        i++;
    }

    return 1;
}


