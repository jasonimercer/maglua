/******************************************************************************
* Copyright (C) 2008-2014 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#ifndef SPINOPERATIONSHORTRANGE
#define SPINOPERATIONSHORTRANGE

#include "spinoperation.h"

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef SHORTRANGE_EXPORTS
  #define SHORTRANGE_API __declspec(dllexport)
 #else
  #define SHORTRANGE_API __declspec(dllimport)
 #endif
#else
 #define SHORTRANGE_API 
#endif

class SHORTRANGE_API ShortRange : public SpinOperation
{
public:
    ShortRange(int nx=32, int ny=32, int nz=1);
    virtual ~ShortRange();
    
    LINEAGE2("ShortRange", "SpinOperation");
    static const luaL_Reg* luaMethods();
    virtual int luaInit(lua_State* L);
    static int help(lua_State* L);
    
    bool apply(SpinSystem* ss);
    
    void addPath(const int site1, const int site2, const double strength, const double* Lab_3x3, const double sig_dot_sig_pow);
    
    virtual void encode(buffer* b);
    virtual int  decode(buffer* b);
    
    typedef struct sss
    {
	int fromsite;
	int tosite;
	double strength;
	double matrix[9];
	double sig_dot_sig_pow;
    } sss;
    
    
    int numPaths() {return num;}
    bool getPath(int idx, int& fx, int& fy, int& fz, int& tx, int& ty, int& tz, double& strength, double* m9, double& sdsp);

    void deinit();
    
    int size;
    int num;
    sss* pathways;
    
    int pbc[3];
};

#endif
