#include "octtree.h"
#include <math.h>
#include "fmm_math.h"


OctTree::OctTree(dArray*  X, dArray*  Y, dArray*  Z,
                 dArray* SX, dArray* SY, dArray* SZ, OctTree* Parent)
{
	inner = 0;
	child_translation_tensor = 0;
    init(X, Y, Z, SX, SY, SZ, Parent);
}

void OctTree::calcLocalOrigin()
{
	for(int i=0; i<3; i++)
		localOrigin[i] = bounds_low[i] + 0.5 * (bounds_high[i] - bounds_low[i]);
}


void OctTree::init(dArray*  X, dArray*  Y, dArray*  Z,
                   dArray* SX, dArray* SY, dArray* SZ, OctTree* Parent)
{
	x = luaT_inc<dArray>(X);
	y = luaT_inc<dArray>(Y);
	z = luaT_inc<dArray>(Z);

	sx = luaT_inc<dArray>(SX);
	sy = luaT_inc<dArray>(SY);
	sz = luaT_inc<dArray>(SZ);

	parent = Parent;

	bounds_low[0] = 0;
	bounds_low[1] = 0;
	bounds_low[2] = 0;

	bounds_high[0] = 1;
	bounds_high[1] = 1;
	bounds_high[2] = 1;

	if(x && y && x && !parent)  //auto generate bounds for root
	{
		dArray& X = *x;
		dArray& Y = *y;
		dArray& Z = *z;

		if(x->nxyz == y->nxyz && y->nxyz == z->nxyz && x->nxyz)
		{
			bounds_low[0] = X[0];
			bounds_low[1] = Y[0];
			bounds_low[2] = Z[0];

			bounds_high[0] = X[0];
			bounds_high[1] = Y[0];
			bounds_high[2] = Z[0];
			for(int i=1; i<x->nxyz; i++)
			{
				if(X[i] < bounds_low[0]) bounds_low[0] = X[i];
				if(Y[i] < bounds_low[1]) bounds_low[1] = Y[i];
				if(Z[i] < bounds_low[2]) bounds_low[2] = Z[i];

				if(X[i] > bounds_high[0]) bounds_high[0] = X[i];
				if(Y[i] > bounds_high[1]) bounds_high[1] = Y[i];
				if(Z[i] > bounds_high[2]) bounds_high[2] = Z[i];
			}

			// expand bounds by a bit (so nodes on domain edges don't get missed on split)
			for(int i=0; i<3; i++)
			{
				double d = bounds_high[i] - bounds_low[i];
				bounds_low[i]  -= d * 0.01;
				bounds_high[i] += d * 0.01;
			}
			calcLocalOrigin();
		}
	}

	for(int i=0; i<8; i++)
	{
		c[i] = 0;
	}

	if(!parent) //this is the root and should have all members
	{
		if(x)
			for(int i=0; i<x->nxyz; i++)
				members.push_back(i);
	}
}

OctTree::~OctTree()
{
	for(int i=0; i<8; i++)
		luaT_dec<OctTree>(c[i]); //dec can deal with null values

    luaT_dec<dArray>(x);
    luaT_dec<dArray>(y);
    luaT_dec<dArray>(z);

    luaT_dec<dArray>(sx);
    luaT_dec<dArray>(sy);
    luaT_dec<dArray>(sz);

	if(inner)
		delete [] inner;
	inner = 0;

	if(child_translation_tensor)
	{
		for(int i=0; i<8; i++)
			delete [] child_translation_tensor[i];
		delete [] child_translation_tensor;
	}
	child_translation_tensor = 0;
}

void OctTree::calcChildTranslationTensorOperator(int max_degree)
{
	if(child_translation_tensor)
	{
		for(int i=0; i<8; i++)
			delete [] child_translation_tensor[i];
		delete [] child_translation_tensor;
	}
	child_translation_tensor = 0;

	if(!c[0])
		return;

	const int tlen = tensor_element_count(max_degree);
	child_translation_tensor = new complex<double>* [8];
	for(int i=0; i<8; i++)
		child_translation_tensor[i] = i2i_trans_mat(max_degree, monopole(localOrigin) - monopole(c[i]->localOrigin));

}

void OctTree::calcInnerTensor(int max_degree, double epsilon)
{
	if(inner)
		delete [] inner;
	inner = 0;
	inner_length = tensor_element_count(max_degree);
	inner = new complex<double>[inner_length];
	for(int i=0; i<inner_length; i++)
	{
		inner[i] = 0;
	}


	if(c[0])
	{
		for(int i=0; i<8; i++)
		{
			c[i]->calcInnerTensor(max_degree);
		}
	}
	else
	{
		dArray& X = *x;
		dArray& Y = *y;
		dArray& Z = *z;

		dArray& SX = *sx;
		dArray& SY = *sy;
		dArray& SZ = *sz;

		monopole origin(localOrigin[0], localOrigin[1], localOrigin[2]);
		complex<double>* temp = new complex<double>[inner_length];

		for(unsigned int i=0; i<members.size(); i++)
		{
			const int j = members[i];

			monopole p(X[j], Y[j], Z[j]);
			monopole m(SX[j], SY[j], SZ[j]);

			const double ml = m.r;
			m.makeUnit();

			monopole pos = p + m * (epsilon / 2.0) - origin;
			pos.q = ml / epsilon;

			InnerTensor(pos, max_degree, temp);

			for(int k=0; k<inner_length; k++)
			{
				inner[k] += temp[k];
			}

			pos = p + m * (epsilon / 2.0) - origin;
			pos.q = -ml / epsilon;

			InnerTensor(pos, max_degree, temp);

			for(int k=0; k<inner_length; k++)
			{
				inner[k] += temp[k];
			}
		}
	}
}


void OctTree::push(lua_State* L)
{
    luaT_push<OctTree>(L, this);
}

int OctTree::luaInit(lua_State* L)
{
    init(luaT_to<dArray>(L, 1), luaT_to<dArray>(L, 2), luaT_to<dArray>(L, 3),
         luaT_to<dArray>(L, 4), luaT_to<dArray>(L, 5), luaT_to<dArray>(L, 6) );

    return 0;
}



int OctTree::help(lua_State* L)
{
    return LuaBaseObject::help(L);
}



bool OctTree::contains(double px, double py, double pz)
{
	if(px < bounds_low[0] || px >= bounds_high[0]) return false;
	if(py < bounds_low[1] || py >= bounds_high[1]) return false;
	if(pz < bounds_low[2] || pz >= bounds_high[2]) return false;
	return true;
}

void OctTree::setBounds(double* low, double* high, int childNumber)
{
	const int a[3] = {1,2,4};
	for(int i=2; i>=0; i--)
	{
		if(childNumber < a[i])
		{
			bounds_low[i] = low[i];
			bounds_high[i] = low[i] + 0.5*(high[i] - low[i]);
		}
		else
		{
			bounds_low[i] = low[i] + 0.5*(high[i] - low[i]);
			bounds_high[i] = high[i];
		}
		childNumber %= a[i];
	}
	calcLocalOrigin();
}


void OctTree::split(int until_contains)
{
	if(c[0]) return; //already split
	
	if((int)members.size() <= until_contains)
		return; //no need to split more
	

	for(int i=0; i<8; i++)
	{
		c[i] = luaT_inc<OctTree>(new OctTree(x, y, z, sx, sy, sz, this));
		c[i]->setBounds(bounds_low, bounds_high, i);
	}

	for(unsigned int i=0; i<members.size(); i++)
	{
		int j = members[i];
		for(int k=0; k<8; k++)
		{
			if(c[k]->contains( (*x)[j],  (*y)[j],  (*z)[j]))
			{
				c[k]->members.push_back(j);
				k = 8; //don't want a node beloning to multiple children (shouldn't happen anyway)
			}
		}
	}
	
	if(until_contains > 0)
	{
		for(int i=0; i<8; i++)
			c[i]->split(until_contains);
	}
}


void OctTree::getStats(double* meanXYZ, double* stddevXYZ)
{
	meanXYZ[0] = 0;
	meanXYZ[1] = 0;
	meanXYZ[2] = 0;
	stddevXYZ[0] = 0;
	stddevXYZ[1] = 0;
	stddevXYZ[2] = 0;
	
	if(!x || members.size() == 0)
		return;
	
	double sum_weight = 0;
	
	for(unsigned int i=0; i<members.size(); i++)
	{
		int j=members[i];
		meanXYZ[0] += (*x)[j];
		meanXYZ[1] += (*y)[j];
		meanXYZ[2] += (*z)[j];
		sum_weight += 1.0;
	}
	if(sum_weight == 0)
		sum_weight = 1.0;
	meanXYZ[0] /= sum_weight;
	meanXYZ[1] /= sum_weight;
	meanXYZ[2] /= sum_weight;
	
	for(unsigned int i=0; i<members.size(); i++)
	{
		int j=members[i];
		stddevXYZ[0] += pow((*x)[j] - meanXYZ[0], 2);
		stddevXYZ[1] += pow((*y)[j] - meanXYZ[1], 2);
		stddevXYZ[2] += pow((*z)[j] - meanXYZ[2], 2);
	}
	
	stddevXYZ[0] /= sum_weight;
	stddevXYZ[1] /= sum_weight;
	stddevXYZ[2] /= sum_weight;
	
	stddevXYZ[0] = sqrt(stddevXYZ[0]);
	stddevXYZ[1] = sqrt(stddevXYZ[1]);
	stddevXYZ[2] = sqrt(stddevXYZ[2]);
}



static int l_split(lua_State* L)
{
    LUA_PREAMBLE(OctTree, oct, 1);
    if(lua_isnumber(L, 2))
        oct->split(lua_tointeger(L, 2));
    else
        oct->split();
    return 0;
}

static int l_child(lua_State* L)
{
    LUA_PREAMBLE(OctTree, oct, 1);

    int c = lua_tointeger(L, 2) - 1;
    if(c >= 0 && c < 8)
    {
        luaT_push<OctTree>(L, oct->c[c]);
    }
    else
    {
        lua_pushnil(L);
    }
    return 1;
}

static int l_tab2array(lua_State* L, int idx, double* a, int n)
{
	if(!lua_istable(L, idx))
		return 0;

	for(int i=1; i<=n; i++)
	{
		lua_pushinteger(L, i);
		lua_gettable(L, idx);
		a[i-1] = lua_tonumber(L, -1);
		lua_pop(L, 1);
	}
	return 1;
}

static void l_array2tab(lua_State* L, double* a, int n)
{
	lua_newtable(L);
	for(int i=1; i<=n; i++)
	{
		lua_pushinteger(L, i);
		lua_pushnumber(L, a[i-1]);
		lua_settable(L, -3);
	}
}

static int l_setbounds(lua_State* L)
{
	LUA_PREAMBLE(OctTree, oct, 1);

	double dummy[3];
	if(l_tab2array(L, 2, dummy, 3))
		memcpy(oct->bounds_low,  dummy, sizeof(double)*3);
	if(l_tab2array(L, 3, dummy, 3))
		memcpy(oct->bounds_high, dummy, sizeof(double)*3);
	return 0;
}


static int l_getbounds(lua_State* L)
{
	LUA_PREAMBLE(OctTree, oct, 1);

	l_array2tab(L, oct->bounds_low, 3);
	l_array2tab(L, oct->bounds_high, 3);

	return 2;
}

static int l_count(lua_State* L)
{
	LUA_PREAMBLE(OctTree, oct, 1);
	lua_pushnumber(L, oct->members.size());
	return 1;
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* OctTree::luaMethods()
{
    if(m[127].name)return m;

    merge_luaL_Reg(m, LuaBaseObject::luaMethods());
    static const luaL_Reg _m[] =
    {
        {"split", l_split},
        {"child", l_child},
		{"setBounds", l_setbounds},
		{"bounds", l_getbounds},
		{"count", l_count},

		//		{"getPosition", l_getpos},
		//		{"preCompute", l_pc},
        {NULL, NULL}
    };
    merge_luaL_Reg(m, _m);
    m[127].name = (char*)1;
    return m;
}










