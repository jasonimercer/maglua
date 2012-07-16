#include "fmm_octtree.h"
#include <math.h>
#include "fmm_math.h"
#include "spinoperation.h"

FMMOctTreeWorkSpace::FMMOctTreeWorkSpace(int max_degree)
{
	tensorA = new complex<double>[tensor_element_count(max_degree)];
	tensorB = new complex<double>[tensor_element_count(max_degree)];
	tensorC = new complex<double>[tensor_element_count(max_degree)];
}

FMMOctTreeWorkSpace::~FMMOctTreeWorkSpace()
{
	delete [] tensorA;
	delete [] tensorB;
	delete [] tensorC;
}


FMMOctTree::FMMOctTree(const int md,
				 dArray*  X, dArray*  Y, dArray*  Z,
				 dArray* SX, dArray* SY, dArray* SZ,
				 dArray* HX, dArray* HY, dArray* HZ,
					   FMMOctTree* Parent)
{
	inner = 0;
	child_translation_tensor = 0;
	init(md, X, Y, Z, SX, SY, SZ, HX, HY, HZ, Parent);
}

void FMMOctTree::calcLocalOrigin()
{
	for(int i=0; i<3; i++)
	{
		bounds_dims[i] = bounds_high[i] - bounds_low[i];
		localOrigin[i] = bounds_low[i] + 0.5 * bounds_dims[i];
	}
}


void FMMOctTree::init(const int md, dArray*  X, dArray*  Y, dArray*  Z,
				   dArray* SX, dArray* SY, dArray* SZ,
				   dArray* HX, dArray* HY, dArray* HZ,
					  FMMOctTree* Parent)
{
	max_degree = md;
	x = luaT_inc<dArray>(X);
	y = luaT_inc<dArray>(Y);
	z = luaT_inc<dArray>(Z);

	sx = luaT_inc<dArray>(SX);
	sy = luaT_inc<dArray>(SY);
	sz = luaT_inc<dArray>(SZ);

	hx = luaT_inc<dArray>(HX);
	hy = luaT_inc<dArray>(HY);
	hz = luaT_inc<dArray>(HZ);

	parent = Parent;

	bounds_low[0] = 0;
	bounds_low[1] = 0;
	bounds_low[2] = 0;

	bounds_high[0] = 1;
	bounds_high[1] = 1;
	bounds_high[2] = 1;

	extra_data = LUA_REFNIL;

	if(!parent)
		generation = 0;
	else
		generation = parent->generation + 1;

	if(!parent)
	{
		WS = new FMMOctTreeWorkSpace(max_degree);
	}
	else
	{
		WS = parent->WS;
	}

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

			// need to make sure hi-low != 0
			for(int i=0; i<3; i++)
			{
				if(fabs(bounds_high[i] - bounds_low[i]) < 1e-8)
				{
					bounds_high[i] += 1;
				}
			}

			if(x->nxyz == 1) //make sure point isn't at r=0
			{
				for(int i=0; i<3; i++)
				{
					bounds_low[i]  -= 0.25;
					bounds_high[i] += 0.5;
				}
			}
			else // expand bounds by a bit (so nodes on domain edges don't get missed on split)
			{
				for(int i=0; i<3; i++)
				{
					double d = bounds_high[i] - bounds_low[i];
					bounds_low[i]  -= d * 0.02; //asymetric
					bounds_high[i] += d * 0.01;
				}
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
			{
				members.push_back(i);
			}
	}
}

FMMOctTree::~FMMOctTree()
{
	for(int i=0; i<8; i++)
		luaT_dec<FMMOctTree>(c[i]); //dec can deal with null values

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

	if(!parent)
		delete WS;
}

int FMMOctTree::totalChildNodes()
{
	int count = 0;
	if(c[0])
	{
		for(int i=0; i<8; i++)
		{
			count++; //child
			count += c[i]->totalChildNodes();
		}
	}
	return count;
}

void FMMOctTree::calcChildTranslationTensorOperators()
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
	{
		monopole r = monopole(c[i]->localOrigin) - monopole(localOrigin);
		child_translation_tensor[i] = i2i_trans_mat(max_degree, -r, new complex<double>[tlen * tlen]);
	}
}

void FMMOctTree::calcInnerTensor(double epsilon)
{
	inner_length = tensor_element_count(max_degree);
	if(!inner)
		inner = new complex<double>[inner_length];

	for(int i=0; i<inner_length; i++)
		inner[i] = 0;


	complex<double>* temp = WS->tensorA;
	if(c[0])
	{

		for(int i=0; i<8; i++)
		{
//			if(c[i]->members.size())
			{
				c[i]->calcInnerTensor(epsilon);

				// need to transform child tensors to local origin
				tensor_mat_mul_LowerTri(child_translation_tensor[i], c[i]->inner, temp, max_degree);

				for(int j=0; j<inner_length; j++)
				{
					inner[j] += temp[j];
				}
			}
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

		monopole origin(localOrigin);

		for(unsigned int i=0; i<members.size(); i++)
		{
			const int j = members[i];

			monopole p(X[j], Y[j], Z[j]);
			monopole m(SX[j], SY[j], SZ[j]);

			const double ml = m.r;
			m.makeUnit();

			// head
			monopole pos = p + m * (epsilon / 2.0) - origin;
			pos.q = ml / epsilon;

			InnerTensor(pos, max_degree, temp);

			for(int k=0; k<inner_length; k++)
			{
				inner[k] += temp[k];
			}

			// tail
			pos = p - m * (epsilon / 2.0) - origin;
			pos.q = -ml / epsilon;

			InnerTensor(pos, max_degree, temp);

			for(int k=0; k<inner_length; k++)
			{
				inner[k] += temp[k];
			}
		}
	}
}


void FMMOctTree::fieldAt(double* p3, double* h3)
{
	int n = tensor_element_count(max_degree);

	complex<double>* dx = WS->tensorA;
	complex<double>* dy = WS->tensorB;
	complex<double>* dz = WS->tensorC;

	gradOutterTensor(monopole(p3) - monopole(localOrigin), max_degree, dx, dy, dz);

	complex<double> _hx = tensor_contract(dx, inner, n);
	complex<double> _hy = tensor_contract(dy, inner, n);
	complex<double> _hz = tensor_contract(dz, inner, n);

	h3[0] = _hx.real();
	h3[1] = _hy.real();
	h3[2] = _hz.real();
}



static void _dipField(const double m[3], const double r[3], double h[3])
{
	double rlen = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
	if(rlen == 0)
	{
		h[0] = 0;
		h[1] = 0;
		h[2] = 0;
	}
	else
	{
		rlen = sqrt(rlen);
		const double ir = 1.0/rlen;
		const double ir3 = ir*ir*ir;
		const double ir5 = ir3*ir*ir;
		const double d = m[0]*r[0] + m[1]*r[1] + m[2]*r[2];
		h[0] = -m[0] * ir3 + 3.0*r[0] * d * ir5;
		h[1] = -m[1] * ir3 + 3.0*r[1] * d * ir5;
		h[2] = -m[2] * ir3 + 3.0*r[2] * d * ir5;
	}
}

void FMMOctTree::calcDipoleFields()
{
	if(c[0])
	{
		if(members.size())
			for(int i=0; i<8; i++)
				c[i]->calcDipoleFields();
		return;
	}

//	dArray& HX = *hx;
//	dArray& HY = *hy;
//	dArray& HZ = *hz;

	dArray& MX = *sx;
	dArray& MY = *sy;
	dArray& MZ = *sz;

	dArray& PX = *x;
	dArray& PY = *y;
	dArray& PZ = *z;

	double mag[3];
	double pos[3];
	double field[3];
	for(unsigned int j=0; j<members.size(); j++)
	{
		int m = members[j];
		double& HX = (*hx)[m];
		double& HY = (*hy)[m];
		double& HZ = (*hz)[m];

		HX = 0;
		HY = 0;
		HZ = 0;

		const double pxm = PX[m];
		const double pym = PY[m];
		const double pzm = PZ[m];

//		iterate over far nodes, cast position in far's coordinates
		for(unsigned int k=0; k<nodes_far.size(); k++)
		{
			const double* o = nodes_far[k]->localOrigin;
			pos[0] = pxm - o[0];
			pos[1] = pym - o[1];
			pos[2] = pzm - o[2];

//			nodes_far[k]->fieldAt(pos, field);

//			HX += field[0];
//			HY += field[1];
//			HZ += field[2];
		}

		for(unsigned int k=0; k<nodes_near.size(); k++)
		{
			const FMMOctTree* n = nodes_near[k];
			for(unsigned int i=0; i<n->members.size(); i++)
			{
				const int q = n->members[i];
				pos[0] = pxm - PX[q];
				pos[1] = pym - PY[q];
				pos[2] = pzm - PZ[q];

				mag[0] = MX[q];
				mag[1] = MY[q];
				mag[2] = MZ[q];

				_dipField(mag, pos, field);

				HX += field[0];
				HY += field[1];
				HZ += field[2];
			}
		}
	}

}


void FMMOctTree::push(lua_State* L)
{
	luaT_push<FMMOctTree>(L, this);
}

int FMMOctTree::luaInit(lua_State* L)
{
	if(!lua_isnumber(L, 1))
	{
		return luaL_error(L, "First argument must be max_degree\n");
	}
	for(int i=2; i<=10; i++)
	{
		if(!luaT_is<dArray>(L, i))
		{
			return luaL_error(L, "Argument %i must be a Double Array\n", i);
		}
	}
	init(   lua_tointeger(L, 1), //max degree
			luaT_to<dArray>(L, 2), luaT_to<dArray>(L, 3), luaT_to<dArray>(L, 4),
			luaT_to<dArray>(L, 5), luaT_to<dArray>(L, 6), luaT_to<dArray>(L, 7),
			luaT_to<dArray>(L, 8), luaT_to<dArray>(L, 9), luaT_to<dArray>(L,10));

    return 0;
}





bool FMMOctTree::contains(double px, double py, double pz)
{
    if(px < bounds_low[0] || px >= bounds_high[0]) return false;
    if(py < bounds_low[1] || py >= bounds_high[1]) return false;
    if(pz < bounds_low[2] || pz >= bounds_high[2]) return false;
    return true;
}

bool FMMOctTree::contains(double* p3)
{
    return contains(p3[0], p3[1], p3[2]);
}

void FMMOctTree::setBounds(double* low, double* high, int childNumber)
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

void FMMOctTree::split(int times)
{
	if(times <= 0)
		return;

	if(c[0])
	{
		for(int i=0; i<8; i++)
		{
			c[i]->split(times-1);
		}
		return; //already split
	}

	for(int i=0; i<8; i++)
	{
		c[i] = luaT_inc<FMMOctTree>(new FMMOctTree(max_degree, x, y, z, sx, sy, sz, hx,hy,hz, this));
		c[i]->setBounds(bounds_low, bounds_high, i);
	}
	calcChildTranslationTensorOperators();

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
	
	split(times);
}


void FMMOctTree::getStats(double* meanXYZ, double* stddevXYZ)
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
	LUA_PREAMBLE(FMMOctTree, oct, 1);
	if(lua_isnumber(L, 2))
		oct->split(lua_tointeger(L, 2));
	else
		oct->split(1);
	return 0;
}

static int l_parent(lua_State* L)
{
	LUA_PREAMBLE(FMMOctTree, oct, 1);
	luaT_push<FMMOctTree>(L, oct->parent);
	return 1;
}

static int l_child(lua_State* L)
{
	LUA_PREAMBLE(FMMOctTree, oct, 1);

	int c = lua_tointeger(L, 2) - 1;
	if(c >= 0 && c < 8)
	{
		luaT_push<FMMOctTree>(L, oct->c[c]);
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
	LUA_PREAMBLE(FMMOctTree, oct, 1);

	double dummy[3];
	if(l_tab2array(L, 2, dummy, 3))
		memcpy(oct->bounds_low,  dummy, sizeof(double)*3);
	if(l_tab2array(L, 3, dummy, 3))
		memcpy(oct->bounds_high, dummy, sizeof(double)*3);
	oct->calcLocalOrigin();
	return 0;
}

static int l_getbounds(lua_State* L)
{
    LUA_PREAMBLE(FMMOctTree, oct, 1);

    l_array2tab(L, oct->bounds_low, 3);
    l_array2tab(L, oct->bounds_high, 3);

    return 2;
}
static int l_getdims(lua_State* L)
{
    LUA_PREAMBLE(FMMOctTree, oct, 1);
    l_array2tab(L, oct->bounds_dims, 3);
    return 1;
}

//static int l_getboundsn(lua_State* L)
//{
//    LUA_PREAMBLE(FMMOctTree, oct, 1);

//    int n=lua_tointeger(L, 2);

//    double d[3];
//    for(int i=0; i<3; i++)
//        d[i] = oct->bounds_dims[i];

//    while(n > 0)
//    {
//        for(int i=0; i<3; i++)
//            d[i] *= 0.5;
//        n--;
//    }
//    while(n < 0)
//    {
//        for(int i=0; i<3; i++)
//            d[i] *= 2.0;
//        n++;
//    }

//    l_array2tab(L, d, 3);
//    return 1;
//}



static int l_count(lua_State* L)
{
	LUA_PREAMBLE(FMMOctTree, oct, 1);
	lua_pushnumber(L, oct->members.size());
	return 1;
}

static int l_calcinnertensor(lua_State* L)
{
	LUA_PREAMBLE(FMMOctTree, oct, 1);
	if(!lua_isnumber(L, 2))
	{
		return luaL_error(L, "`calcInnerTensor' requires an epsilon");
	}
	double epsilon = lua_tonumber(L, 2);
	oct->calcInnerTensor(epsilon);
	return 0;
}

static int l_fieldAt(lua_State* L)
{
	LUA_PREAMBLE(FMMOctTree, oct, 1);
	double pos[3];
	double field[3];

	int r1 = lua_getNdouble(L, 3, pos, 2, 0);

	oct->fieldAt(pos, field);

	lua_pushnumber(L, field[0]);
	lua_pushnumber(L, field[1]);
	lua_pushnumber(L, field[2]);

	return 3;
}


static int l_maxdegree(lua_State* L)
{
	LUA_PREAMBLE(FMMOctTree, oct, 1);
	lua_pushnumber(L, oct->max_degree);
	return 1;
}

static int l_getlocalorigin(lua_State* L)
{
	LUA_PREAMBLE(FMMOctTree, oct, 1);

	l_array2tab(L, oct->localOrigin, 3);
	return 1;
}

static int l_getinnertensor(lua_State* L)
{
	LUA_PREAMBLE(FMMOctTree, oct, 1);

	const int n = tensor_element_count(oct->max_degree);
	const complex<double>* t = oct->inner;

	if(!t)
		lua_pushnil(L);
	else
	{
		lua_newtable(L);
		for(int i=0; i<n; i++)
		{
			lua_pushinteger(L, i+1);
			lua_newtable(L);

			lua_pushinteger(L, 1);
			lua_pushnumber(L, t[i].real());
			lua_settable(L, -3);
			lua_pushinteger(L, 2);
			lua_pushnumber(L, t[i].imag());
			lua_settable(L, -3);

			lua_settable(L, -3);
		}
	}

	return 1;
}


static int l_getmember(lua_State* L)
{
	LUA_PREAMBLE(FMMOctTree, oct, 1);

	int i = lua_tointeger(L, 2);

	if(i<0 || i>oct->members.size())
		return 0;

	lua_pushinteger(L, oct->members[i-1]+1);
	return 1;
}

static int l_childtranstens(lua_State* L)
{
	LUA_PREAMBLE(FMMOctTree, oct, 1);

	int c = lua_tointeger(L, 2)-1;

	if(c<0 || c>=8)
		return luaL_error(L, "Child index out of bounds");

	if(!oct->child_translation_tensor || !oct->child_translation_tensor[c])
		return luaL_error(L, "child tensor is null");

	const int n = oct->inner_length;

	lua_newtable(L);
	for(int i=0; i<n; i++)
	{
		lua_pushinteger(L, i+1);

		lua_newtable(L);
		for(int j=0; j<n; j++)
		{
			complex<double> v = oct->child_translation_tensor[c][i*n+j];
			lua_pushinteger(L, j+1);

			lua_newtable(L);
			lua_pushinteger(L, 1);
			lua_pushnumber(L, v.real());
			lua_settable(L, -3);
			lua_pushinteger(L, 2);
			lua_pushnumber(L, v.imag());
			lua_settable(L, -3);

			lua_settable(L, -3);
		}
		lua_settable(L, -3);
	}

	return 1;
}

static int l_totalchildnodes(lua_State* L)
{
	LUA_PREAMBLE(FMMOctTree, oct, 1);

	lua_pushinteger(L, oct->totalChildNodes());
	return 1;
}

static int l_getgeneration(lua_State* L)
{
	LUA_PREAMBLE(FMMOctTree, oct, 1);

	lua_pushinteger(L, oct->generation);
	return 1;
}


static int l_if(lua_State* L)
{
	double pos[3];
	int r1 = lua_getNdouble(L, 3, pos, 2, 0);

	monopole m(pos);
	int n = lua_tointeger(L, 2+r1);
	int l = lua_tointeger(L, 2+r1+1);
	luaT< complex<double> >::push(L, Inner(m, n, l));
	return luaT< complex<double> >::elements();
}


static int l_getextradata(lua_State* L)
{
	LUA_PREAMBLE(FMMOctTree, oct, 1);

	if(oct->extra_data < 0)
		lua_pushnil(L);
	else
		lua_rawgeti(L, LUA_REGISTRYINDEX, oct->extra_data);
	return 1;
}

static int l_setextradata(lua_State* L)
{
	LUA_PREAMBLE(FMMOctTree, oct, 1);

	if(oct->extra_data != LUA_REFNIL)
	{
		luaL_unref(L, LUA_REGISTRYINDEX, oct->extra_data);
	}

	lua_pushvalue(L, 2);

	oct->extra_data = luaL_ref(L, LUA_REGISTRYINDEX);

	return 0;
}


static int l_near(lua_State* L)
{
	LUA_PREAMBLE(FMMOctTree, p1, 1);
	LUA_PREAMBLE(FMMOctTree, p2, 2);

	if(p1->generation != p2->generation)
	{
		lua_pushboolean(L, 0);
		return 1;
	}

	const double* dims = p1->bounds_dims;

	for(int i=0; i<3; i++)
	{
		if( fabs(p1->localOrigin[i] - p2->localOrigin[i]) * 1.01 > dims[i])
		{
			lua_pushboolean(L, 0);
			return 1;
		}

	}

	lua_pushboolean(L, 1);
	return 1;
}
static int l_eq(lua_State* L)
{
    LUA_PREAMBLE(FMMOctTree, p1, 1);
    LUA_PREAMBLE(FMMOctTree, p2, 2);
    lua_pushboolean(L, p1==p2);
    return 1;
}

static int l_lt(lua_State* L)
{
    LUA_PREAMBLE(FMMOctTree, p1, 1);
    LUA_PREAMBLE(FMMOctTree, p2, 2);
    lua_pushboolean(L, p1<p2);
    return 1;
}
static int l_le(lua_State* L)
{
    LUA_PREAMBLE(FMMOctTree, p1, 1);
    LUA_PREAMBLE(FMMOctTree, p2, 2);
    lua_pushboolean(L, p1<=p2);
    return 1;
}



static int l_addnear(lua_State* L)
{
	LUA_PREAMBLE(FMMOctTree, p1, 1);
	LUA_PREAMBLE(FMMOctTree, p2, 2);
	p1->nodes_near.push_back(p2);
	return 0;
}

static int l_addfar(lua_State* L)
{
	LUA_PREAMBLE(FMMOctTree, p1, 1);
	LUA_PREAMBLE(FMMOctTree, p2, 2);
	p1->nodes_far.push_back(p2);
	return 0;
}


static int l_uid(lua_State* L)
{
    LUA_PREAMBLE(FMMOctTree, p1, 1);
    lua_pushinteger(L, (lua_Integer)p1);
    return 1;
}

static int l_calcdipfield(lua_State* L)
{
	LUA_PREAMBLE(FMMOctTree, p1, 1);
	p1->calcDipoleFields();
	return 0;
}


static int l_contains(lua_State* L)
{
    LUA_PREAMBLE(FMMOctTree, p1, 1);

    if(luaT_is<FMMOctTree>(L, 2))
    {
        LUA_PREAMBLE(FMMOctTree, p2, 2);
        lua_pushboolean(L, p1->contains(p2->localOrigin));
        return 1;
    }

    double pos[3];
    if(lua_getNdouble(L, 3, pos, 2, 0) < 0)
    {
        return luaL_error(L, "Failed to parse position");
    }

    lua_pushboolean(L, p1->contains(pos));
    return 1;


}

int FMMOctTree::help(lua_State* L)
{
    if(lua_gettop(L) == 0)
    {
        lua_pushstring(L, "The main data structure used in the *Multipole* operator");
        lua_pushstring(L, "1 Integer, 9 Double Arrays: The integer is the maximum degree used in the spherical harmonic tensors, the double arrays come in triplets denoting XYZ components of: Position, Spin Orientation and Output Field.");
        lua_pushstring(L, ""); //output, empty
        return 3;
    }



    if(!lua_isfunction(L, 1))
    {
        return luaL_error(L, "help expect zero arguments or 1 function.");
    }

    lua_CFunction func = lua_tocfunction(L, 1);

	if(func == l_split)
	{
		lua_pushstring(L, "Subdivide the FMMOctTree Node");
		lua_pushstring(L, "1 or 0 Integers: number of times to recersively split, default 0");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_parent)
	{
		lua_pushstring(L, "Get the parent of the node");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 FMMOctTree Node: The Parent");
		return 3;
	}

	if(func == l_child)
	{
		lua_pushstring(L, "Get a child of the node");
		lua_pushstring(L, "1 Integer: The index of the desired child. 1 to 8.");
		lua_pushstring(L, "1 FMMOctTree Node: The Child");
		return 3;
	}

	if(func == l_setbounds)
	{
		lua_pushstring(L, "Manually set the bounds of a node. This is not required but useful for testing purposes.");
		lua_pushstring(L, "2 Tables of 3 Numbers: The new bounds for the node {minx, miny, minz}, {maxx, maxy, maxz}. The local origin will be updated to be the center of the volume.");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_getbounds)
	{
		lua_pushstring(L, "Get the bounds of a node.");
		lua_pushstring(L, "");
		lua_pushstring(L, "2 Tables of 3 Numbers: The bounds for the node {minx, miny, minz}, {maxx, maxy, maxz}.");
		return 3;
	}

	if(func == l_getdims)
	{
		lua_pushstring(L, "Get the edge lengths of the node.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Tables of 3 Numbers: The edge lengths of the node in each cartesian direction.");
		return 3;
	}

	if(func == l_count)
	{
		lua_pushstring(L, "Get the number of points contained in this node.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: The number of points contained in the volume of this node");
		return 3;
	}


	if(func == l_calcinnertensor)
	{
		lua_pushstring(L, "Update Inner Tensor for this node. If it is a leaf node it is computed from the contained points otherwise it recursively computes the Inner Tensor of the children and combines their tensors after translation of origins.");
		lua_pushstring(L, "1 Number: Epsilon. This is the \"infinitesimal\" separation between a dipole's positive and negative charge. A number such as 1e-5 may work.");
		lua_pushstring(L, "");
		return 3;
	}


	if(func == l_fieldAt)
	{
		lua_pushstring(L, "Calculate the field at a point relative to the local origin due to the calculated Inner Tensor. The given point must be farther than the origin than all contained points.");
		lua_pushstring(L, "1 *3Vector*: The x,y,z coordinates of the field sample point relative to the local origin");
		lua_pushstring(L, "3 Numbers: The field's x, y and z components");
		return 3;
	}

	if(func == l_maxdegree)
	{
		lua_pushstring(L, "Query the maximum degree used in internal spherical harmonic expansions.");
		lua_pushstring(L, "");
		lua_pushstring(L, "3 Integer: The maximum degree");
		return 3;
	}

	if(func == l_getinnertensor)
	{
		lua_pushstring(L, "Get the computed Inner Tensor of the node");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Table of real/imag pairs: The Inner Tensor in the internal format");
		return 3;
	}

	if(func == l_getlocalorigin)
	{
		lua_pushstring(L, "Get the center of the volume.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Table: The x,y and z coordinates of the center of the volume.");
		return 3;
	}


	if(func == l_getmember)
	{
		lua_pushstring(L, "Lookup the index in the main arrays for a contained point.");
		lua_pushstring(L, "1 Integer: The local index for a contained point.");
		lua_pushstring(L, "1 Integer: The global index of a contained point.");
		return 3;
	}

	if(func == l_childtranstens)
	{
		lua_pushstring(L, "Get the translation matrix used to translate a child Inner Tensor.");
		lua_pushstring(L, "1 Integer: The child index.");
		lua_pushstring(L, "1 Table: A table of tables representing a complex matrix.");
		return 3;
	}

	if(func == l_contains)
	{
		lua_pushstring(L, "Test if a given point falls inside the node's volume.");
		lua_pushstring(L, "1 *3Vector*: The test point.");
		lua_pushstring(L, "1 Boolean: True if inside volume, otherwise false.");
		return 3;
	}


	if(func == l_setextradata)
	{
		lua_pushstring(L, "Set extra lua data for a node");
		lua_pushstring(L, "1 Data: Something to store in the node");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_getextradata)
	{
		lua_pushstring(L, "Get extra lua data for a node");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Data: Something stored in the node");
		return 3;
	}



//	{"totalChildNodes", l_totalchildnodes},
//	{"generation", l_getgeneration},
//	{"uid", l_uid},



//	{"innerFunction", l_if},
//	{"near", l_near},


//    if(func == l_mappos)
//    {
//        lua_pushstring(L, "Map lattice coordinates to arbitrary positions in space");
//        lua_pushstring(L, "1 *3Vector* (Integers), 1 *3Vector* (Numbers), [1 Number]: Index of site to map from, position in space to map to, optional weight of data (default 1.0).");
//        lua_pushstring(L, "");
//        return 3;
//    }


    return LuaBaseObject::help(L);
}



static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* FMMOctTree::luaMethods()
{
    if(m[127].name)return m;

    merge_luaL_Reg(m, LuaBaseObject::luaMethods());
    static const luaL_Reg _m[] =
    {
		{"split", l_split},
		{"parent", l_parent},
		{"child", l_child},
//		{"children", l_children},
		{"setBounds", l_setbounds},
        {"bounds", l_getbounds},
        //{"boundsOfNChild", l_getboundsn},
        {"dimensions", l_getdims},
        {"count", l_count},
		{"calcInnerTensor", l_calcinnertensor},
		{"fieldAt", l_fieldAt},
		{"maxDegree", l_maxdegree},
		{"innerTensor", l_getinnertensor},
		{"localOrigin", l_getlocalorigin},
		{"member", l_getmember},
		{"childTranslationTensor", l_childtranstens},
		{"totalChildNodes", l_totalchildnodes},
        {"generation", l_getgeneration},
        {"uid", l_uid},
        {"contains", l_contains},

		{"setExtraData", l_setextradata},
		{"extraData",    l_getextradata},

		{"addNear", l_addnear},
		{"addFar",  l_addfar},

		{"calculateDipoleFields", l_calcdipfield},

		{"innerFunction", l_if},
		{"near", l_near},
        {"__eq", l_eq},
        {"__lt", l_lt},
        {"__le", l_le},
        {NULL, NULL}
    };
    merge_luaL_Reg(m, _m);
    m[127].name = (char*)1;
    return m;
}










