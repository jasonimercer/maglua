/******************************************************************************
* Copyright (C) 2008-2011 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#ifndef INTERPOLATINGFUCTION_H
#define INTERPOLATINGFUCTION_H

#include "luacommon.h"
#include <vector>
#include "encodable.h"

using namespace std;

class CORE_API InterpolatingFunction : public Encodable
{
public:
	InterpolatingFunction();
	~InterpolatingFunction();

	void addData(const double in, const double out);
	bool getValue(double in, double* out);
	int refcount;	

	void encode(buffer* b) const;
	int  decode(buffer* b);
	
private:
	class _node
	{
	public:
		_node(double x1, double y1, double x2, double y2);
		_node(_node* c0, _node* c1);
		~_node();

		bool inrange(const double test);

		_node* c[2];
		double x[2];
		double y[2];
		double m, cut;
	};
	
	void compile();

	bool compiled;

	vector <pair<double,double> > rawdata;
	_node* root;
};

CORE_API InterpolatingFunction* checkInterpolatingFunction(lua_State* L, int idx);
CORE_API void registerInterpolatingFunction(lua_State* L);
CORE_API void lua_pushInterpolatingFunction(lua_State* L, InterpolatingFunction* if1D);

// 
#endif

