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

#ifndef SPINOPERATIONANISOTROPY
#define SPINOPERATIONANISOTROPY

#include "spinoperation.h"

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef ANISOTROPY_EXPORTS
  #define ANISOTROPY_API __declspec(dllexport)
 #else
  #define ANISOTROPY_API __declspec(dllimport)
 #endif
#else
 #define ANISOTROPY_API 
#endif

class Anisotropy : public SpinOperation
{
public:
	Anisotropy(int nx=32, int ny=32, int nz=1);
	virtual ~Anisotropy();
	
	LINEAGE2("Anisotropy", "SpinOperation")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	static int help(lua_State* L);
	
	bool apply(SpinSystem* ss);
	void addAnisotropy(int site, double nx, double ny, double nz, double K);

	bool getAnisotropy(int site, double& nx, double& ny, double& nz, double& K);
	
	typedef struct ani
	{
		int site;
		double axis[3];
		double strength;
	} ani;
	
	ani* ops;
	int size;
	int num;
	
	int merge();
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);

	void init();
	void deinit();
};


#endif
