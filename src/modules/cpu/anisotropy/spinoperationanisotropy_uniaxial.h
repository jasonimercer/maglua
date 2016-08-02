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

#ifndef SPINOPERATIONANISOTROPYUNIAXIAL
#define SPINOPERATIONANISOTROPYUNIAXIAL

#include "spinoperation.h"

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef ANISOTROPYUNIAXIAL_EXPORTS
  #define ANISOTROPYUNIAXIAL_API __declspec(dllexport)
 #else
  #define ANISOTROPYUNIAXIAL_API __declspec(dllimport)
 #endif
#else
 #define ANISOTROPYUNIAXIAL_API 
#endif

class AnisotropyUniaxial : public SpinOperation
{
public:
	AnisotropyUniaxial(int nx=32, int ny=32, int nz=1);
	virtual ~AnisotropyUniaxial();
	
	LINEAGE2("Anisotropy.Uniaxial", "SpinOperation")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);
	
	bool apply(SpinSystem* ss);
	void addAnisotropy(int site, double* axis, double K1, double K2=0);
	bool getAnisotropy(int site, double* axis, double& K1, double& K2);
	
	typedef struct ani
	{
		int site;
		double axis[3];
		double K[2];
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
