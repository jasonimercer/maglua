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

#include "luacommon.h"
#include "dipolesupport.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "extrapolate.h"
#include <vector>
using namespace std;

#ifdef WIN32
 #include <windows.h>
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)
 #pragma warning(disable: 4996)
 #define snprintf _snprintf
#endif



#define ZERO_CHECK double r = sqrt(rx*rx+ry*ry+rz*rz); if(r < 1E-10)	return 0; double ir = 1.0 / r;

double gamma_xx_dip(double rx, double ry, double rz)
{
	ZERO_CHECK
	return - pow(ir,3) + 3.0 * rx * rx * pow(ir,5);
}

double gamma_xy_dip(double rx, double ry, double rz)
{
	ZERO_CHECK
	return           + 3.0 * rx * ry * pow(ir,5);
}

double gamma_xz_dip(double rx, double ry, double rz)
{
	ZERO_CHECK
	return           + 3.0 * rx * rz * pow(ir,5);
}

double gamma_yy_dip(double rx, double ry, double rz)
{
	ZERO_CHECK
	return - pow(ir,3) + 3.0 * ry * ry * pow(ir,5);
}

double gamma_yz_dip(double rx, double ry, double rz)
{
	ZERO_CHECK
	return           + 3.0 * ry * rz * pow(ir,5);
}

double gamma_zz_dip(double rx, double ry, double rz)
{
	ZERO_CHECK
	return - pow(ir,3) + 3.0 * rz * rz * pow(ir,5);
}


static void getWAB_range(const double* ABC, 
	const int nA, const int nB, const int nC,  //width, depth, layers 
	const int ix, const int iy, const int iz,
	const int ax, const int ay, const int bx, const int by,	      
	              const int truemax, 
	double& gXX, double& gXY, double& gXZ,
	double& gYY, double& gYZ, double& gZZ)
{
    {
	for(int x=ax; x<=bx; x++)
	{
	    const int xx = x*nA+ix;
	    
	    if(abs(xx) <= truemax)
	    {
		for(int y=ay; y<=by; y++)
		{
		    const int yy = y*nB+iy;
		    if(abs(yy) <= truemax)
		    {
			const double rx = ((double)xx)*ABC[0] + ((double)yy)*ABC[3] + ((double)iz)*ABC[6];
			const double ry = ((double)xx)*ABC[1] + ((double)yy)*ABC[4] + ((double)iz)*ABC[7];
			const double rz = ((double)xx)*ABC[2] + ((double)yy)*ABC[5] + ((double)iz)*ABC[8];
			
			const double r2 = rx*rx + ry*ry + rz*rz;
			if(r2 != 0)
			{
			    const double ir  = 1.0/sqrt(r2);
			    const double ir3 = ir*ir*ir;
			    const double ir5 = ir3*ir*ir;
			    
			    gXX += ir3 - 3.0 * rx * rx * ir5;
			    gXY +=     - 3.0 * rx * ry * ir5;
			    gXZ +=     - 3.0 * rx * rz * ir5;
			    
			    gYY += ir3 - 3.0 * ry * ry * ir5;
			    gYZ +=     - 3.0 * ry * rz * ir5;
							
			    gZZ += ir3 - 3.0 * rz * rz * ir5;
			}
		    }
		}
	    }
	}
    }
}


///periodic XY
static void getWAB(
	const double* ABC, 
	const int nA, const int nB, const int nC,  //width, depth, layers 
	const int ix, const int iy, const int iz, 
	const int smin, const int smax, const int truemax, //allowing rings 
	double* XX, double* XY, double* XZ,
	double* YY, double* YZ, double* ZZ)
{
	double gXX = 0;
	double gXY = 0;
	double gXZ = 0;
	double gYY = 0;
	double gYZ = 0;
	double gZZ = 0;

	const int minx = smin;
	const int miny = smin;
	const int maxx = smax;
	const int maxy = smax;

	// +----b
	// |    |
	// |    |
	// a----+

	// each coordinate here denotes a lattice
	int ax[9];
	int ay[9];
	int bx[9];
	int by[9];

	ax[0] = -(maxx-1);
	ay[0] = miny;
	bx[0] = -minx;
	by[0] = (maxy-1);

	ax[1] = -minx+1;
	ay[1] = ay[0];
	bx[1] = minx-1;
	by[1] = by[0];

	ax[2] = minx;
	ay[2] = ay[0];
	bx[2] = maxx-1;
	by[2] = by[0];

	ax[3] = ax[0];
	ay[3] = -(miny-1);
	bx[3] = bx[0];
	by[3] = miny-1;

	ax[4] = ax[2];
	ay[4] = ay[3];
	bx[4] = bx[2];
	by[4] = by[3];

	ax[5] = ax[0];
	ay[5] = -(maxy-1);
	bx[5] = bx[0];
	by[5] = -miny;

	ax[6] = ax[1];
	ay[6] = ay[5];
	bx[6] = bx[1];
	by[6] = by[5];

	ax[7] = ax[2];
	ay[7] = ay[5];
	bx[7] = bx[2];
	by[7] = by[5];

	ax[8] = -(maxx-1);
	ay[8] = -(maxy-1);
	bx[8] = -ax[8];
	by[8] = -ay[8];

	if(smin == 0)
	{
	    const int i = 8;
	    getWAB_range(ABC, nA, nB, nC, ix, iy, iz,
			 ax[i], ay[i], bx[i], by[i],
			 truemax, 
			 gXX, gXY, gXZ,
			 gYY, gYZ, gZZ);
	}
	else
	{
	    for(int i=0; i<8; i++)
	    {
		getWAB_range(ABC, nA, nB, nC, ix, iy, iz,
			     ax[i], ay[i], bx[i], by[i],
			     truemax,
			     gXX, gXY, gXZ,
			     gYY, gYZ, gZZ);
	    }
	}

	*XX = -gXX;
	*XY = -gXY;
	*XZ = -gXZ;

	*YY = -gYY;
	*YZ = -gYZ;

	*ZZ = -gZZ;
}//end function WAB


static int _isZeroMat(const double* M, int nx, int ny)
{
	for(int j=0; j<ny; j++)
	{
		for(int i=0; i<nx; i++)
		{
			if(fabs(M[i+j*nx]) > 1E-16)
				return 0;
		}
	}
	return 1;
}

static void _writemat(FILE* f, const char* name, int zoffset, const double* M, int nx, int ny)
{
	fprintf(f, "\n");
	
	if(_isZeroMat(M, nx, ny))
	{
		fprintf(f, "%s[%i] = 0\n", name, zoffset);
		return;
	}
	
	fprintf(f, "%s[%i] = [[\n", name, zoffset);

	for(int j=0; j<ny; j++)
	{
// 		fprintf(f, "    {");
		for(int i=0; i<nx; i++)
		{
// 			fprintf(f, "% 12e%s", M[i+j*nx], i==(nx-1)?"}":", ");
			if(M[i+j*nx] == 0)
			{
				fprintf(f, "0%s", (i==(nx-1) && j==(ny-1))?"":",");
			}
			else
			{
				fprintf(f, "% .16g%s", M[i+j*nx], (i==(nx-1) && j==(ny-1))?"":",");
			}
		}
		fprintf(f, "\n");
// 		fprintf(f, "%c\n", j==(ny-1)?' ':',');
	}
	fprintf(f, "]]\n");
	fprintf(f, "\n");
}

static void _writeParser(FILE* f)
{
	const char* parse =
		"\n"
		"function tokenizeNumbers(line)\n"
		"	local t = {}\n"
		"	for w in string.gmatch(line, \"[^,]+\") do\n"
		"		table.insert(t, tonumber(w))\n"
		"	end\n"
		"	return t\n"
		"end\n"
		"\n"
		"function tokenizeLines(lines)\n"
		"	-- strip empty lines\n"
		"	lines = string.gsub(lines, \"^%s*\\n*\", \"\")\n"
		"	lines = string.gsub(lines, \"\\n\\n+\", \"\\n\")\n"
		"	\n"
		"	local t = {}\n"
		"	for w in string.gmatch(lines, \"(.-)\\n\" ) do\n"
		"		table.insert(t, tokenizeNumbers(w))\n"
		"	end\n"
		"	\n"
		"	return t\n"
		"end\n"
		"\n"
		"function parseMatrix(M)\n"
		"	if M == 0 then\n"
		"		-- returns a 2D table that always returns zero\n"
		"		local tz, ttz = {}, {}\n"
		"		setmetatable(tz,  {__index = function() return  0 end})\n"
		"		setmetatable(ttz, {__index = function() return tz end})\n"
		"		return ttz\n"
		"	end\n"
		"	\n"
		"	return tokenizeLines(M)\n"
		"end\n"
		"\n"
		"function map(f, t)\n"
		"	for k,v in pairs(t) do\n"
		"		t[k] = f(v)\n"
		"	end\n"
		"	return t\n"
		"end\n"
		"\n"
		"function parse()\n"
		"	XX = map(parseMatrix, XX)\n"
		"	XY = map(parseMatrix, XY)\n"
		"	XZ = map(parseMatrix, XZ)\n"
		"\n"
		"	YY = map(parseMatrix, YY)\n"
		"	YZ = map(parseMatrix, YZ)\n"
		"\n"
		"	ZZ = map(parseMatrix, ZZ)\n"
		"end\n";

	fprintf(f, "%s", parse);
}

static bool dipole_write_matrix(const char* filename,
	const double* ABC,
	const int nx, const int ny, const int nz,  //width, depth, layers 
	const int gmax, 
	const double* XX, const double* XY, const double* XZ,
	const double* YY, const double* YZ, const double* ZZ)
{
	FILE* f = fopen(filename, "w");
	if(!f)
		return false;
	
	fprintf(f, "-- This file contains dipole interaction matrices\n");
	fprintf(f, "\n");
	if(gmax == -1)
		fprintf(f, "gmax = math.huge\n");
	else
		fprintf(f, "gmax = %i\n", gmax);
		
	fprintf(f, "nx, ny, nz = %i, %i, %i\n", nx, ny, nz);
	fprintf(f, "ABC = {{%g, %g, %g}, --unit cell\n       {%g, %g, %g},\n       {%g, %g, %g}}\n\n", 
		ABC[0], ABC[1], ABC[2],
		ABC[3], ABC[4], ABC[5],
		ABC[6], ABC[7], ABC[8]);
	fprintf(f, "XX={} XY={} XZ={} YY={} YZ={} ZZ={}\n");
	
	int c = 0;
	for(int zoffset=0; zoffset<nz; zoffset++)
	{
		_writemat(f, "XX", zoffset, &XX[c*nx*ny], nx, ny);
		_writemat(f, "XY", zoffset, &XY[c*nx*ny], nx, ny);
		_writemat(f, "XZ", zoffset, &XZ[c*nx*ny], nx, ny);
		
		_writemat(f, "YY", zoffset, &YY[c*nx*ny], nx, ny);
		_writemat(f, "YZ", zoffset, &YZ[c*nx*ny], nx, ny);
		
		_writemat(f, "ZZ", zoffset, &ZZ[c*nx*ny], nx, ny);
		
		c++;
	}
	
	
	_writeParser(f);
	
	fclose(f);
	return true;
}

static void next_dipolefilename(const char* current, char* next, int len, const int nx, const int ny)
{
	if(current && current[0])
	{
		int x, y, v;
		sscanf(current, "WAB_%ix%i.%i.lua", &x, &y, &v);
		snprintf(next, len, "WAB_%ix%i.%i.lua", x, y, v+1);
	}
	else
	{
		snprintf(next, len, "WAB_%ix%i.%i.lua", nx, ny, 1);
	}
}

static int file_exists(const char* filename)
{
	FILE* f = fopen(filename, "r");
	if(f)
	{
		fclose(f);
		return 1;
	}
	return 0;
}


static bool valueMatch(lua_State* L, const char* name, int value)
{
	lua_getglobal(L, name);
	if(!lua_isnumber(L, -1))
	{
		lua_pop(L, 1);
		return false;
	}
	
	int v = lua_tointeger(L, -1);
	lua_pop(L, 1);
	return v == value;
}

static bool approxSame(double a, double b)
{
	bool c = fabs(a-b) <= 0.5*(fabs(a) + fabs(b)) * 1e-6;
	return c;
}

static bool checkTable(lua_State* L, const double* v3)
{
	if(!lua_istable(L, -1))
	{
		return false;
	}
	for(int i=0; i<3; i++)
	{
		lua_pushinteger(L, i+1);
		lua_gettable(L, -2);
		if(!lua_isnumber(L, -1) || !(approxSame(lua_tonumber(L, -1), v3[i])))
		{
			lua_pop(L, 1);
			return false;
		}
		lua_pop(L, 1);
	}
	return true;
}

static bool dipoleParamsMatch(
	const char* filename,
	const int nx, const int ny, const int nz,
	const int gmax, double* ABC)
{
	lua_State *L = lua_open();
	luaL_openlibs(L);
	
	if(luaL_dofile(L, filename))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		lua_close(L);
		return false;
	}
	
	const char* nn[3] = {"nx", "ny", "nz"};
	int  nv[3]; 
	nv[0] = nx; nv[1] = ny; 
	nv[2] = nz;

	for(int i=0; i<3; i++)
	{
		if(!valueMatch(L, nn[i], nv[i]))
		{
			lua_close(L);
			return false;
		}
	}
	
	int file_gmax = 0;
	lua_getglobal(L, "gmax");
	file_gmax = lua_tointeger(L, -1);
	lua_getglobal(L, "math");
	lua_pushstring(L, "huge");
	lua_gettable(L, -2);
	lua_remove(L, -2); //remove math table
	if(lua_equal(L, -2, -1)) //then gmax = math.huge
	{
		file_gmax = -1; //special marker for math.huge
	}
	lua_pop(L, 2);
	
	if(file_gmax != gmax)
	{
		lua_close(L);
		return false;
	}
	
	//see if unit cell matches
	lua_getglobal(L, "ABC");
	if(!lua_istable(L, -1))
	{
		lua_close(L);
		return false;
	}
	
	for(int i=0; i<3; i++)
	{
		lua_pushinteger(L, i+1);
		lua_gettable(L, -2); // get A/B/C
		if(!checkTable(L, ABC+3*i))
		{
			lua_close(L);
			return false;
		}
		lua_pop(L, 1);
	}
	lua_close(L);
	return true;
}

static void loadXYZ(
	const char* filename,
	const int nx, const int ny, const int nz,
	double* XX, double* XY, double* XZ,
	double* YY, double* YZ, double* ZZ)
{
	lua_State* L = lua_open();
	luaL_openlibs(L);
	
	const char* vars[6] = {"XX", "XY", "XZ", "YY", "YZ", "ZZ"};
	double* arrs[6];
	arrs[0] = XX;
	arrs[1] = XY;
	arrs[2] = XZ;
	arrs[3] = YY;
	arrs[4] = YZ;
	arrs[5] = ZZ;
	
	if(luaL_dofile(L, filename))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		lua_close(L);
		return;
	}
	
	lua_getglobal(L, "parse");
	if(lua_isfunction(L, -1))
	{
	  if(lua_pcall(L, 0, 0, 0))
	    {
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		lua_close(L);
		return;	      
	    }
	}

	const int nxyz = nx*ny*nz;
	for(int a=0; a<6; a++)
	{
		int c = 0;
		//int p = 0;
		lua_getglobal(L, vars[a]); //XX
		for(int k=0; k<nz; k++)
		{
			lua_pushinteger(L, k); //XX 0
			lua_gettable(L, -2);   //XX XX[0]
			for(int j=0; j<ny; j++)
			{
				lua_pushinteger(L, j+1); // XX XX[0] 1
				lua_gettable(L, -2);     // XX XX[0] XX[0,1]
				for(int i=0; i<nx; i++)
				{
					lua_pushinteger(L, i+1); // XX XX[0] XX[0,1] 2
					lua_gettable(L, -2);     // XX XX[0] XX[0,1] XX[0,1,2]
					arrs[a][c*nx*ny + j*nx + i] = lua_tonumber(L, -1);
					lua_pop(L, 1); // XX XX[0] XX[0,1]
				}
				lua_pop(L, 1); // XX XX[0]
			}
			lua_pop(L, 1); // XX
			c++;
		}
		lua_pop(L, 1); //
	}
	
	lua_close(L);
}

static bool extrapolate(lua_State* L,
			vector<int>& cuttoffs,
            vector<double>& vXX, vector<double>& vXY, vector<double>& vXZ, 
                                 vector<double>& vYY, vector<double>& vYZ, 
                                                      vector<double>& vZZ, bool& fail)
{
	vector< vector<double> > vAB;
	vAB.push_back(vXX);
	vAB.push_back(vXY);
	vAB.push_back(vXZ);
	vAB.push_back(vYY);
	vAB.push_back(vYZ);
	vAB.push_back(vZZ);
	
	double sol[6];
	bool   res[6];
	bool   ok = true;

	for(int i=0; i<6 && ok; i++)
	{
		lua_pop(L, lua_gettop(L));
		lua_getglobal(L, "extrapolate");
		lua_newtable(L);
		for(unsigned int j=0; j<cuttoffs.size(); j++)
		{
			lua_pushinteger(L, j+1);
			lua_newtable(L); //{x,y} holder
			lua_pushinteger(L, 1);
			lua_pushinteger(L, cuttoffs[j]);
			lua_settable(L, -3); //set x
			lua_pushinteger(L, 2);
			lua_pushnumber(L, vAB[i][j]);
			lua_settable(L, -3); // set y
			lua_settable(L, -3); // add {x,y} pair
		}
		
		if(lua_pcall(L, 1, 2, 0))
		{
			fprintf(stderr, "ERROR IN HARDCODED EXTRAPOLATE SCRIPT:\n  %s\n", lua_tostring(L, -1));
			exit(1);
			fail = true;
			return false;
		}
		
		int b = lua_toboolean(L, -1);
		if(b)
		{
			sol[i] = lua_tonumber(L, -2);
		}
		res[i] = b;

		lua_pop(L, 2);

		ok &= (bool)b;
		//printf("%i %i\n", i, b);
	}
	
	if(ok)
	{
		fail = false;
		vXX.push_back(sol[0]);
		vXY.push_back(sol[1]);
		vXZ.push_back(sol[2]);
		vYY.push_back(sol[3]);
		vYZ.push_back(sol[4]);
		vZZ.push_back(sol[5]);
		return true;
	}
	return false;
}


void dipoleLoad(
	const int nx, const int ny, const int nz,
	const int gmax, double* ABC,
	double* XX, double* XY, double* XZ,
	double* YY, double* YZ, double* ZZ)
{
	char fn[1024] = ""; //mmm arbitrary
	
	lua_State* L = lua_open();
	luaL_openlibs(L);
	
	if(luaL_dostring(L, __extrapolate))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		lua_close(L);
		return;
	}
	
	while(true)
	{
		next_dipolefilename(fn, fn, 1024, nx, ny);
		if(file_exists(fn))
		{
			if(dipoleParamsMatch(fn, nx, ny, nz, gmax, ABC))
			{
				loadXYZ(fn, 
						nx, ny, nz,
						XX, XY, XZ,
						YY, YZ, ZZ);
				lua_close(L);
				return;
			}
		}
		else
		{
			int c = 0;
			for(int k=0; k<nz; k++)
			for(int j=0; j<ny; j++)
			for(int i=0; i<nx; i++)
			{
				//printf("get %i %i %i\n", i,j,k);
				fflush(stdout);
				if(gmax != -1)
				{
					getWAB(ABC,
						nx, ny, nz,
						i, j, k,
						0, gmax, gmax,
						XX+c, XY+c, XZ+c,
						YY+c, YZ+c, ZZ+c);
				}
				else // math.huge sum
				{
					vector<double> vXX;
					vector<double> vXY;
					vector<double> vXZ;
					vector<double> vYY;
					vector<double> vYZ;
					vector<double> vZZ;
					vector<int> cuttoffs;
					
					double tXX, tXY, tXZ, tYY, tYZ, tZZ;
					double sXX, sXY, sXZ, sYY, sYZ, sZZ;
					sXX=0; sXY=0; sXZ=0;
					sYY=0; sYZ=0; sZZ=0;

					const int lstep = 5; //these are in steps of lattices
					bool fail = false;
					int _lmin = 0; //inits of lattices
					int _lmax = lstep;
					
					bool converge = false;
					int q = 0;
					int maxiter = 1000;
					do
					{
						getWAB(ABC, nx,ny,nz, i,j,k, _lmin, _lmax, (int)1e16, &tXX, &tXY, &tXZ, &tYY, &tYZ, &tZZ);
						sXX+=tXX;sXY+=tXY;sXZ+=tXZ;
						sYY+=tYY;sYZ+=tYZ;sZZ+=tZZ;
						vXX.push_back(sXX);
						vXY.push_back(sXY);
						vXZ.push_back(sXZ);
						vYY.push_back(sYY);
						vYZ.push_back(sYZ);
						vZZ.push_back(sZZ);

						cuttoffs.push_back(_lmax);
						_lmin += lstep;
						_lmax += lstep;
						if(q>=8) //let the system prime itself before trying to extrapolate
						{
						    converge = extrapolate(L, cuttoffs, vXX, vXY, vXZ, vYY, vYZ, vZZ, fail);
							//q -= 4; //collect more data before trying again
						}
						q++;
						maxiter--;
					}while(!converge && !fail && maxiter);
				
// 					if(converge || fail)
					{
					    int last = vXX.size() - 1;
					    
					    XX[c] = vXX[last];
					    XY[c] = vXY[last];
					    XZ[c] = vXZ[last];

					    YY[c] = vYY[last];
					    YZ[c] = vYZ[last];

					    ZZ[c] = vZZ[last];
					}
	
					if(fail | !maxiter)
					{
					    fprintf(stderr, "Failed to find a extrapolate to solution under tolerance, using best calculated value\n");
					}
				}
				c++;
			}


			dipole_write_matrix(fn,
				ABC,
				nx, ny, nz,
				gmax,
				XX, XY, XZ,
				YY, YZ, ZZ);
			lua_close(L);
			return;
		}
	}
	lua_close(L);
	printf("done %i %i %i\n", nx, ny, nz);
}

