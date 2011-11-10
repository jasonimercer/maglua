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
#include "magnetostaticssupport.h"
#include "gamma_ab_v.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef WIN32
 #include <windows.h>
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)
 #pragma warning(disable: 4996)
 #define snprintf _snprintf
#endif

#ifndef M_PI
#define M_PI 3.14159265358979
#endif
/*#define DipMagConversion (1.0 / (M_PI * 4.0))*/
#define DipMagConversion (1.0)

#include "../dipole_cuda/dipolesupport.h" //to compare for crossover

typedef struct mag_dip_crossover
{
	double r2;  //crossover at this distance (squared) (initially big)
	double tol; //crossover at this tolerance (reset r)
} mag_dip_crossover;

bool equal_tol(double a, double b, double tolerance)
{
// 	if(a == b)
// 		return true;
	if(a == 0 && b == 0)
		return true;
	
	if(b == 0 || a == 0)
		return false;
	
	double t = fabs( (a-b)/b );
// 	printf("%E, %e - %f, %f\n", a, b, t, tolerance);
	return t < tolerance;
}

double min(double a, double b)
{
	if(a<b)
		return a;
	return b;
}

///periodic XY
static void getGAB(
	const double* ABC, 
	const double* prism, /* 3 vector */
	const int nA, const int nB, const int nC,  //width, depth, layers 
	const int ix, const int iy, const int iz, 
	const int gmax, 
	double* XX, double* XY, double* XZ,
	double* YY, double* YZ, double* ZZ,
	mag_dip_crossover& crossover)
{
	int smax;
	int i, j, c;
	double r2;
	double ir,ir3,ir5;
	double rx, ry, rz;
	int iL;

	if(nA < nB)
		iL = nA;
	else
		iL = nB;

	smax = gmax/iL + 1;

	double volume2 = pow(prism[0] * prism[1] * prism[2], 2);
	
	const double d1 = prism[0];
	const double l1 = prism[1];
	const double w1 = prism[2];
	
	const double d2 = prism[0];
	const double l2 = prism[1];
	const double w2 = prism[2];
	
	double* gXX = new double[smax*2+1];
	double* gXY = new double[smax*2+1];
	double* gXZ = new double[smax*2+1];
	double* gYY = new double[smax*2+1];
	double* gYZ = new double[smax*2+1];
	double* gZZ = new double[smax*2+1];

	double magXX, magXY, magXZ;
	double dipXX, dipXY, dipXZ;
	
	double magYY, magYZ;
	double dipYY, dipYZ;
	
	double magZZ;
	double dipZZ;
	
	for(c=0; c<smax*2+1; c++)
	{
		gXX[c] = 0;
		gXY[c] = 0;
		gXZ[c] = 0;
		gYY[c] = 0;
		gYZ[c] = 0;
		gZZ[c] = 0;
	}

	/* sum over periodic lattices */
	for(j=-smax; j<= smax; j++)
		for(i=-smax, c=0; i<= smax; i++, c++)
		{
			const int xx = i*nA+ix;
			const int yy = j*nB+iy;
			const int zz = iz;

			if(abs(xx) <= gmax && abs(yy) <= gmax && abs(zz) <= gmax)
			{
				rx = ((double)i*nA+ix)*ABC[0] + ((double)j*nB+iy)*ABC[3] + ((double)iz)*ABC[6];
				ry = ((double)i*nA+ix)*ABC[1] + ((double)j*nB+iy)*ABC[4] + ((double)iz)*ABC[7];
				rz = ((double)i*nA+ix)*ABC[2] + ((double)j*nB+iy)*ABC[5] + ((double)iz)*ABC[8];
				r2 = rx*rx + ry*ry + rz*rz;
				//if(r2 != 0)

				if(r2 >= crossover.r2)
				{

					gXX[c] += DipMagConversion * volume2 * gamma_xx_dip(rx, ry, rz);
					gXY[c] += DipMagConversion * volume2 * gamma_xy_dip(rx, ry, rz);
					gXZ[c] += DipMagConversion * volume2 * gamma_xz_dip(rx, ry, rz);

					gYY[c] += DipMagConversion * volume2 * gamma_yy_dip(rx, ry, rz);
					gYZ[c] += DipMagConversion * volume2 * gamma_yz_dip(rx, ry, rz);
					gZZ[c] += DipMagConversion * volume2 * gamma_zz_dip(rx, ry, rz);
				}
				else
				{
					if(xx == 0 && yy == 0 && zz == 0)
					{
						magXX = gamma_xx_sv(d1, l1, w1)*2*M_PI * -4.0;
						magYY = gamma_yy_sv(d1, l1, w1)*2*M_PI * -4.0;
						magZZ = gamma_zz_sv(d1, l1, w1)*2*M_PI * -4.0;
					}
					else
					{
						magXX = gamma_xx_v(rx, ry, rz, d1, l1, w1, d2, l2, w2);
						magYY = gamma_yy_v(rx, ry, rz, d1, l1, w1, d2, l2, w2);
						magZZ = gamma_zz_v(rx, ry, rz, d1, l1, w1, d2, l2, w2);
					}

					magXY = gamma_xy_v(rx, ry, rz, d1, l1, w1, d2, l2, w2);
					magXZ = gamma_xz_v(rx, ry, rz, d1, l1, w1, d2, l2, w2);
					magYZ = gamma_yz_v(rx, ry, rz, d1, l1, w1, d2, l2, w2);

					dipXX = DipMagConversion * volume2 * gamma_xx_dip(rx, ry, rz);
					dipXY = DipMagConversion * volume2 * gamma_xy_dip(rx, ry, rz);
					dipXZ = DipMagConversion * volume2 * gamma_xz_dip(rx, ry, rz);

					dipYY = DipMagConversion * volume2 * gamma_yy_dip(rx, ry, rz);
					dipYZ = DipMagConversion * volume2 * gamma_yz_dip(rx, ry, rz);
					dipZZ = DipMagConversion * volume2 * gamma_zz_dip(rx, ry, rz);

					// 				if( sqrt(rx*rx+ry*ry+rz+rz) > 40)
					// 					printf("md: %E, %E\n", magXX, dipXX);

					bool same = equal_tol(magXX, dipXX, crossover.tol) &&
						equal_tol(magXY, dipXY, crossover.tol) &&
						equal_tol(magXZ, dipXZ, crossover.tol) &&
						equal_tol(magYY, dipYY, crossover.tol) &&
						equal_tol(magYZ, dipYZ, crossover.tol) &&
						equal_tol(magZZ, dipZZ, crossover.tol);

					if(same && r2 > 0)
					{
						crossover.r2 = min(crossover.r2, fabs(r2));

						//printf("crossover at: r = %f\n", sqrt(crossover.r2));
					}

					gXX[c] += magXX;
					gXY[c] += magXY;
					gXZ[c] += magXZ;

					gYY[c] += magYY;
					gYZ[c] += magYZ;
					gZZ[c] += magZZ;	
				}
#ifndef WIN32
#warning This is a hack to fix self terms. Eventually this will be in the numerical code.
#endif
				if(xx == 0 && yy == 0 && zz == 0)
				{
					gXX[c] *= 0.5;
					gXY[c] *= 0.5;
					gXZ[c] *= 0.5;

					gYY[c] *= 0.5;
					gYZ[c] *= 0.5;

					gZZ[c] *= 0.5;
				}
			}
		}
	
	*XX = 0;
	for(c=0; c<smax*2+1; c++)
		*XX += gXX[c];

	*XY = 0;
	for(c=0; c<smax*2+1; c++)
		*XY += gXY[c];

	*XZ = 0;
	for(c=0; c<smax*2+1; c++)
		*XZ += gXZ[c];

	*YY = 0;
	for(c=0; c<smax*2+1; c++)
		*YY += gYY[c];

	*YZ = 0;
	for(c=0; c<smax*2+1; c++)
		*YZ += gYZ[c];

	*ZZ = 0;
	for(c=0; c<smax*2+1; c++)
		*ZZ += gZZ[c];

	
	delete [] gXX;
	delete [] gXY;
	delete [] gXZ;

	delete [] gYY;
	delete [] gYZ;

	delete [] gZZ;
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
			fprintf(f, "% 12e%s", M[i+j*nx], (i==(nx-1) && j==(ny-1))?"":",");
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

static bool magnetostatics_write_matrix(const char* filename,
	const double* ABC,
	const double* prism,
	const int nx, const int ny, const int nz,  //width, depth, layers 
	const int gmax, 
	const double* XX, const double* XY, const double* XZ,
	const double* YY, const double* YZ, const double* ZZ)
{
	FILE* f = fopen(filename, "w");
	if(!f)
		return false;
	
	fprintf(f, "-- This file contains magnetostatic interaction matrices\n");
	fprintf(f, "\n");
	fprintf(f, "gmax = %i\n", gmax);
	fprintf(f, "nx, ny, nz = %i, %i, %i\n", nx, ny, nz);
	fprintf(f, "cellDimensions = {%g, %g, %g}\n", prism[0], prism[1], prism[2]);
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

static void next_magnetostaticsfilename(const char* current, char* next, int len, const int nx, const int ny)
{
	if(current && current[0])
	{
		int x, y, v;
		sscanf(current, "GAB_%ix%i.%i.lua", &x, &y, &v);
		snprintf(next, len, "GAB_%ix%i.%i.lua", x, y, v+1);
	}
	else
	{
		snprintf(next, len, "GAB_%ix%i.%i.lua", nx, ny, 1);
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

static bool checkTable(lua_State* L, const double* v3)
{
	if(!lua_istable(L, -1))
		return false;
	for(int i=0; i<3; i++)
	{
		lua_pushinteger(L, i+1);
		lua_gettable(L, -2);
		if(!lua_isnumber(L, -1) || (lua_tonumber(L, -1) != v3[i]))
		{
			lua_pop(L, 1);
			return false;
		}
		lua_pop(L, 1);
	}
	return true;
}

static bool magnetostaticsParamsMatch(
	const char* filename,
	const int nx, const int ny, const int nz,
	const int gmax, const double* ABC, const double* prism)
{
	lua_State *L = lua_open();
	luaL_openlibs(L);
	
	if(luaL_dofile(L, filename))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		lua_close(L);
		return false;
	}
	
	const char* nn[4] = {"nx", "ny", "nz", "gmax"};
	int  nv[4]; 
	nv[0] = nx; nv[1] = ny; 
	nv[2] = nz; nv[3] = gmax;

	for(int i=0; i<4; i++)
	{
		if(!valueMatch(L, nn[i], nv[i]))
		{
			lua_close(L);
			return false;
		}
	}
	
	lua_getglobal(L, "cellDimensions");
	if(!checkTable(L, prism))
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

void magnetostaticsLoad(
	const int nx, const int ny, const int nz,
	const int gmax, const double* ABC,
	const double* prism, /* 3 vector */
	double* XX, double* XY, double* XZ,
	double* YY, double* YZ, double* ZZ,
	double tol)
{
	char fn[64] = "";
	
	mag_dip_crossover crossover;
	crossover.r2  = 1E10;
	crossover.tol = tol;
	
	
	while(true)
	{
		next_magnetostaticsfilename(fn, fn, 64, nx, ny);
		if(file_exists(fn))
		{
			if(magnetostaticsParamsMatch(fn, nx, ny, nz, gmax, ABC, prism))
			{
				loadXYZ(fn, 
						nx, ny, nz,
						XX, XY, XZ,
						YY, YZ, ZZ);
				return;
			}
		}
		else
		{
			int c = 0;
			for(int k=0; k<nz; k++)
			{
				for(int j=0; j<ny; j++)
				{
					for(int i=0; i<nx; i++)
					{
						getGAB(ABC,
							prism,
							nx, ny, nz,
							i, j, k,
							gmax,
							XX+c, XY+c, XZ+c,
							YY+c, YZ+c, ZZ+c, 
							crossover);
						c++;
					}
				}
			}

			magnetostatics_write_matrix(fn,
				ABC, prism,
				nx, ny, nz,
				gmax,
				XX, XY, XZ,
				YY, YZ, ZZ);
			return;
		}
	}
}


