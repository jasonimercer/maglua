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

#ifndef INTERPOLATINGFUCTION2D_H
#define INTERPOLATINGFUCTION2D_H

#include "luacommon.h"
#include <vector>
#include "encodable.h"

using namespace std;

class InterpolatingFunction2D : public Encodable
{
public:
	InterpolatingFunction2D();
	~InterpolatingFunction2D();

	void addData(const double inx, const double iny, const double out);
	bool getValue(double inx, double iny, double* out);
	bool compile();
	int refcount;

	void setInvalidValue(const double d);
		
	double xmin,  ymin; 
	double xmax,  ymax; 

	bool compiled;
	bool hasInvalidValue;
	double invalidValue;
	
	void encode(buffer* b);
	int  decode(buffer* b);
	
private:
	class triple
	{
	public:
		triple(double X=0, double Y=0, double Z=0) : x(X), y(Y), z(Z) {};
		double x, y, z;
	};

	void getixiy(double x, double y, int *v2);

	int getidx(double x, double y);


	double xstep, ystep;
	int nx, ny;

	double* data;

	vector < triple > rawdata;
};

COREPURECUDA_API InterpolatingFunction2D* checkInterpolatingFunction2D(lua_State* L, int idx);
COREPURECUDA_API void registerInterpolatingFunction2D(lua_State* L);
COREPURECUDA_API void lua_pushInterpolatingFunction2D(lua_State* L, Encodable* if2D);
// 
#endif

