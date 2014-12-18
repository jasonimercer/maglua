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

typedef struct bfnode
{
    vint idx; // index into spheres
    vint neighbour; // neighbour index in flattened vec2idx style 
    double max_path_energy;
    double energy;
    int source; 
    int distance;
}bfnode;

static void vint_overlap(vint& a, vint& b, vint& c)
{
    std::map<int,int> mm;

    for(int i=0; i<a.size(); i++)
        mm[a[i]]++;
    for(int i=0; i<b.size(); i++)
        mm[b[i]]++;
    
    c.clear();
    for(std::map<int,int>::iterator it=mm.begin(); it != mm.end(); ++it)
    {
        if(it->second > 1)
            c.push_back(it->first);
    }
}

static int vec2idx(vint& vec, vint max)
{
    int j = 0;
    int c = 1;
    for(int i=0; i<max.size(); i++)
    {
        j += vec[i] * c;
        c *= max[i];
    }
    return j;
}

static void idx2vec(int idx, vint& vec, vint max)
{
    vec.clear();
    vec.resize(max.size(), 0);

    for(int i=max.size()-1; i>=0; i--)
    {
        int c = 1;
        for(int j=0; j<i; j++)
            c *= max[j];

        vec[i] = idx / c;
        idx -= vec[i] * c;
    }
}

static double max_(double a, double b)
{
    if(a > b)
        return a;
    return b;
}

static void vint_print(vint& v)
{
    for(int i=0; i<v.size(); i++)
        printf("%d\n", v[i]);
}

static bool refineList(list<int>& x, bfnode* nodes, const int total_nodes)
{
//    printf("refine\n");
    
    bool change = false;
    
    vector<int> a;
    vector<int> b;
    vector<int> c;
    vector<int> common;
    
    for(list<int>::iterator i=x.begin(); i!=x.end(); ++i)
        a.push_back(*i);
    
    
    b.push_back(a[0]);
    for(int i=0; i<a.size()-2; i++)
    {
        /*
        printf("a[%i].neighbours:\n", i);
        vint_print(nodes[a[i]].neighbour);
        printf("a[%i].neighbours:\n", i+2);
        vint_print(nodes[a[i+2]].neighbour);
       
        printf("a[%i] = %i\n", i+1, a[i+1]);
        */

        vint_overlap(nodes[a[i]].neighbour, nodes[a[i+2]].neighbour, common);

        //printf("common:\n");
        //vint_print(common);

        
        int smallest = 0;
        for(int j=1; j<common.size(); j++)
        {
            if(nodes[common[j]].energy < nodes[common[smallest]].energy)
                smallest = j;
        }
        
        if(common[smallest] != a[i+1])
        {
            change = true;
            // printf("changing a[i+1] from %i to %i\n", a[i+1], common[smallest]);
        }
        b.push_back(common[smallest]);
    }
    b.push_back(a.back());
    
    
    c.push_back(b[0]);
    for(int i=1; i<b.size(); i++)
    {
        int& x = c.back();
        if(x != b[i])
            c.push_back(b[i]);
        else
        {
            // printf("removing duplicate %i\n", b[i]);
            change = true;
        }
    }

    x.clear();
    for(int i=0; i<c.size(); i++)
        x.push_back(c[i]);
    
    return change;
}



int l_bestpath(lua_State* L) //Bellman-Ford path search
{
    LUA_PREAMBLE(MEP, mep, 1);
    int ns = mep->numberOfSites();

    if(!lua_istable(L, 2) || !lua_istable(L, 3))
    {
        return luaL_error(L, "Path needs start and end.");
    }

    vector<VectorCS> a;
    vector<VectorCS> b;

    lua_pushnil(L);
    while(lua_next(L, 2))
    {
        a.push_back(lua_toVectorCS(L, lua_gettop(L)));
        lua_pop(L, 1);
    }

    lua_pushnil(L);
    while(lua_next(L, 3))
    {
        b.push_back(lua_toVectorCS(L, lua_gettop(L)));
        lua_pop(L, 1);
    }

    vector<double> mags;

    for(int i=0; i<a.size(); i++)
        mags.push_back(a[i].magnitude());


    int n = 1; // sphere level
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

    vector<int> sphere_verts;
    int total_nodes = 1;
    for(int i=0; i<ns; i++)
    {
        sphere_verts.push_back(0);

        while(spheres[i][sphere_verts[i]].neighbours)
        {
            sphere_verts[i]++;
        }
        total_nodes*= sphere_verts[i];
    }

    int last_idx  = vec2idx(b_idx, sphere_verts);
    int first_idx = vec2idx(a_idx, sphere_verts);

    bfnode* nodes = new bfnode[total_nodes];

    /*
    {
        vint test;
        idx2vec(33, test, sphere_verts);

        
        printf("%d   %d %d     %d %d     \n", 33, test[0], test[1], sphere_verts[0], sphere_verts[1]);

        int* x = (int*)5;
        *x =4;
    }*/
    
    for(int i=0; i<total_nodes; i++)
    {
        idx2vec(i, nodes[i].idx, sphere_verts);
        //printf("%d   %d %d     %d %d     \n", i, nodes[i].idx[0], nodes[i].idx[1], sphere_verts[0], sphere_verts[1]);

        vint num_neighbours;
        vector<const int*> neighbours;
        for(int j=0; j<ns; j++)
        {
            const int* nn = spheres[j][nodes[i].idx[j]].neighbours;
            neighbours.push_back(nn);

            int k=0;
            while(nn[k] != -1)
                k++;
            num_neighbours.push_back(k);
        }

        vint state;
        for(int j=0; j<ns; j++)
            state.push_back(0);

        vint vec;
        vec.resize(ns, 0);
        do
        {
            for(int j=0; j<ns; j++)
            {
                vec[j] = neighbours[j][ state[j] ];
            }
            int flat = vec2idx(vec, sphere_verts);

            //printf("%d %d\n", vec[0], vec[1]);
            nodes[i].neighbour.push_back(flat);
        }while(v_inc(state, num_neighbours));

        vector<VectorCS> cfg;

        for(int j=0; j<ns; j++)
        {
            VectorCS v(spheres[j][ nodes[i].idx[j] ].vertex);
            v.setMagnitude(mags[j]);
            cfg.push_back(v);
        }

        nodes[i].energy = mep->energyOfCustomPoint(cfg);
        nodes[i].max_path_energy = INFINITY;
        nodes[i].distance = 1e8; // should be OK
        nodes[i].source = i;

        if(i == first_idx)
        {
            nodes[i].max_path_energy = nodes[i].energy;
            nodes[i].distance = 0;
            nodes[i].source = i;
        }
    }

    // we are now setup to run iterations of the Bellman-Ford algorithm

    /*
    vint idx; // index into spheres
    vint neighbour; // neighbour index in flattened vec2idx style 
    double max_path_energy;
    double energy;
    int source;
    int distance;
    */

    bool progress = true;

    int iteration = 0;

    while(progress)
    {
        progress = false;
        iteration++;
        //printf("it %d\n", iteration);

        for(int i=0; i<total_nodes; i++)
        {
            if(i != first_idx)
            {
                const int s = nodes[i].source;
                int best_path = nodes[i].neighbour[0];

                for(int j=1; j<nodes[i].neighbour.size(); j++)
                {
                    const int k = nodes[i].neighbour[j];
                    
                    if(nodes[k].max_path_energy == nodes[best_path].max_path_energy)
                    {
                        if(nodes[k].distance == nodes[best_path].distance)
                        {
                            if(nodes[k].energy < nodes[best_path].energy)
                            {
                                best_path = k;
                            }
                        }

                        if(nodes[k].distance < nodes[best_path].distance)
                        {
                            best_path = k;
                        }
                    }
                    
                    if(nodes[k].max_path_energy < nodes[best_path].max_path_energy)
                    {
                        best_path = k;
                    }
                }
                
                
                if(best_path != nodes[i].source)
                {
                    progress = true;
                    
                    const double ee = max_(nodes[i].energy, nodes[best_path].max_path_energy);
                    
                    nodes[i].max_path_energy = ee;
                    nodes[i].distance = nodes[best_path].distance + 1;
                    nodes[i].source = best_path;
                }
            }
        }
    }



    list<int> soln_flat;

    int path = 0;
    while(last_idx != first_idx && last_idx >= 0)
    {
        // printf("last_idx = %d\n", last_idx);
        /*
        vint x;
        idx2vec(last_idx, x, sphere_verts);
        soln.push_front(x);
        */

        soln_flat.push_front(last_idx);

        if(nodes[last_idx].source == last_idx)
            last_idx = -1;
        else
            last_idx = nodes[last_idx].source;
    }
    soln_flat.push_front(first_idx);

    /// ABC
    while(refineList(soln_flat, nodes, total_nodes));
    delete [] nodes;


    
    list<vint> soln;

    for(list<int>::iterator it=soln_flat.begin(); it!=soln_flat.end(); ++it)
    {
        vint x;
        idx2vec(*it, x, sphere_verts);

        soln.push_back( vint(x));
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

#if 1

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
        
        int max = 0;
        while(srcs[i][max] != -1)
        {
            max++;
        }
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
            distance = 0;
        }
    dijkstrasNode(vint& _here)
        : here(_here)
        {
            source_energy = INFINITY; 
            energy_calculated=false;
            considered = false;
            distance = 0;
        }
    dijkstrasNode(const dijkstrasNode& n)
        {
            energy_calculated = n.energy_calculated;
            energy = n.energy;
            source_energy = n.source_energy;
            source = n.source;
            considered = n.considered;
            neighbour_list = n.neighbour_list;
            distance = n.distance;
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

            /*
            printf("here: %d %d\n", here[0], here[1]);

            printf("Neighbours:\n");
            for(int i=0; i<neighbour_list.size(); i++)
                printf("%d %d\n", neighbour_list[i][0], neighbour_list[i][1]);
            {int* i=(int*)1; *i=1;}
            */
        }
    ;

    double maxPathEnergy(std::map<vint, dijkstrasNode>& nodes)
        {
            if(vint_same(here, source))
                return energy;

            return max_(nodes[here].maxPathEnergy(nodes), energy);
        }
    

    bool energy_calculated;
    double energy;
    double source_energy;
    vint source;
    int distance;
    vint here;

    vector<vint> neighbour_list;

    bool considered;

};

class dijkstrasSearch
{
public:
    dijkstrasSearch() {};

    vint getSource(vint& n)
        {
            return nodes[n].source;
        }

    double getEnergy(vint& n)
        {
            return nodes[n].energy;
        }

    void commonNeighbours(const vint& a, const vint& b, vector<vint>& common)
        {
            common.clear();
            std::map<vint, int> nmap; // neighbour map
            
            vector<vint>& na = nodes[a].neighbour_list;
            vector<vint>& nb = nodes[b].neighbour_list;

            for(int i=0; i<na.size(); i++)
                nmap[na[i]]++;

            for(int i=0; i<nb.size(); i++)
                nmap[nb[i]]++;

            std::map<vint, int>::iterator it;
            for(it = nmap.begin(); it != nmap.end(); ++it)
            {
                if(it->second > 1)
                    common.push_back(vint(it->first));
            }
        }

    bool refineList(list<vint>& x)
        {
            printf("refine\n");

            bool change = false;

            vector<vint> a;
            vector<vint> b;
            vector<vint> c;
            vector<vint> common;

            for(list<vint>::iterator i=x.begin(); i!=x.end(); ++i)
                a.push_back(*i);

            
            b.push_back(a[0]);
            for(int i=0; i<a.size()-2; i++)
            {
                commonNeighbours(a[i], a[i+2], common);

                //printf("Neighbours of [%d %d] and [%d %d]:\n", a[i][0], a[i][1], a[i+2][0], a[i+2][1]);
                //printf("Between: [%d %d]\n", a[i+1][0], a[i+1][1]);

                int smallest = 0;
                for(int j=0; j<common.size(); j++)
                {
                    if(getEnergy(common[j]) < getEnergy(common[smallest]))
                        smallest = j;

                    //printf("[%d %d]\n", common[j+1][0], common[j+1][1]);
                    
                }

                if(!vint_same(common[smallest], a[i+1]))
                {
                    change = true;
                    
                    //printf("Replacing [%d %d %5e] with [%d %d %5e]\n", a[i+1][0], a[i+1][1],getEnergy(a[i+1]), common[smallest][0], common[smallest][1], getEnergy(common[smallest]));
                }
                b.push_back(common[smallest]);
            }
            b.push_back(a.back());


            c.push_back(b[0]);
            for(int i=1; i<b.size(); i++)
            {
                vint& x = c.back();
                if(!vint_same(x, b[i]))
                    c.push_back(b[i]);
                else
                {
                    //printf("dropping duplicate\n");
                    change = true;
                }
            }

            x.clear();
            for(int i=0; i<c.size(); i++)
                x.push_back(c[i]);

            return change;
        }


    bool iterate(vint& source)
        {
            if(edge.empty())
            {
                return false;
            }

            vint here = edge.front();
            edge.pop_front();


            if( nodes.find(here) == nodes.end() ) // then doesn't contain
            {
                nodes.insert ( std::pair<vint,dijkstrasNode>(here,dijkstrasNode(here)) );
                nodes[here].create_neighbour_list(here, spheres);
                if(vint_same(here, source))
                {
                    nodes[here].distance = 0;
                }
            }

            dijkstrasNode& node = nodes.find(here)->second;

            if(node.considered)
            {
                return true;
            }


            double energy_here = node.getEnergy(mep, here, spheres);

            for(int i=0; i<node.neighbour_list.size(); i++)
            {
                vint& nn = node.neighbour_list[i];

                if( nodes.find(nn) == nodes.end() ) // then doesn't contain
                {
                    nodes.insert ( std::pair<vint,dijkstrasNode>(nn,dijkstrasNode(nn)) );
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
                        neighbour.source = vint(here);
                    }
                    edge.push_back(nn);
                }
            }
            node.considered = true;
            return true;
        };

    
    std::map<vint, dijkstrasNode> nodes;
    std::list<vint> edge;

    vector<const sphere*> spheres;    
    MEP* mep;
};

int l_bestpath2(lua_State* L) // dijkstra's implementation
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
        lua_pop(L, 1);
    }

    lua_pushnil(L);
    while(lua_next(L, 3))
    {
        b.push_back(lua_toVectorCS(L, lua_gettop(L)));
        lua_pop(L, 1);
    }

    if(a.size() != b.size())
        return luaL_error(L, "Configuration size mismatch");

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
    while(ds.iterate(a_idx))
    {
        //printf("dsi %d\n", dsi);
        //dsi++;
    }

    list<vint> soln;

    soln.push_front(vint(b_idx));
    
    int path = 0;
    while(!vint_same(b_idx, a_idx))
    {
        //printf("path %d\n", path);
        //path++;
        b_idx = vint(ds.getSource(b_idx));
        soln.push_front(vint(b_idx));
    }

    printf("soln size %d\n", soln.size());

    while( ds.refineList(soln) );

    printf("soln size %d\n", soln.size());

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


#endif
