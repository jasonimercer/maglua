/******************************************************************************
* Copyright (C) 2008-2012 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#ifndef BSPTREE
#define BSPTREE

#include "luabaseobject.h"

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef BSPTREE_EXPORTS
  #define BSPTREE_API __declspec(dllexport)
 #else
  #define BSPTREE_API __declspec(dllimport)
 #endif
#else
 #define BSPTREE_API 
#endif




#include <vector>
using namespace std;

class BSPData
{
public:	
	BSPData(double x, double y, double z, int _ref)
	{
		p[0] = x;
		p[1] = y;
		p[2] = z;
		ref = _ref;
	}
	double p[3];
	int ref;
};

class BSPDataList
{
public:	
	vector<BSPData*> data;
	
	bool getXYZv(int idx, double* p3)
	{
		if(idx < 0 || idx >= data.size())
			return false;
		p3[0] = data[idx]->p[0];
		p3[1] = data[idx]->p[1];
		p3[2] = data[idx]->p[2];
		return true;
	}
};



class BSPTREE_API BSPTree : public LuaBaseObject
{
public:
	BSPTree();
	~BSPTree();

	LINEAGE1("BSPTree")
	static const luaL_Reg* luaMethods();
	virtual void push(lua_State* L);
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);
	
	void encode(buffer* b);
	int decode(buffer* b);

	
	int addData(const double x, const double y, const double z, const int ref);
	int addIndex(const int i);
	
	int getDataInSphere(const double cx, const double cy, const double cz, const double r, int& next_id);

	void init();
	void deinit();
		
	void setList(BSPDataList* l);
	double p1[3], p2[3]; //aabb

	BSPTree* c[2];
	
	BSPDataList* list;
	vector<int> idx;
	
	int split_dir;
	void split(int max_at_leaf = 1);
	
	BSPTree* parent;
};

#endif
