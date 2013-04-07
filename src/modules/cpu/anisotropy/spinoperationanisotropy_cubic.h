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

#ifndef SPINOPERATIONANISOTROPYCUBIC
#define SPINOPERATIONANISOTROPYCUBIC

#include "spinoperation.h"

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef ANISOTROPYCUBIC_EXPORTS
  #define ANISOTROPYCUBIC_API __declspec(dllexport)
 #else
  #define ANISOTROPYCUBIC_API __declspec(dllimport)
 #endif
#else
 #define ANISOTROPYCUBIC_API 
#endif

class AnisotropyCubic : public SpinOperation
{
public:
	AnisotropyCubic(int nx=32, int ny=32, int nz=1);
	virtual ~AnisotropyCubic();
	
	LINEAGE2("Anisotropy.Cubic", "SpinOperation")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);
	
	virtual const char* getSlotName();
		
	bool apply(SpinSystem* ss);
	void addAnisotropy(int site, double* a1, double* a2, double* K3);

	bool getAnisotropy(int site, double* a1, double* a2, double* a3, double* K3);
	
	typedef struct ani
	{
		int site;
		double axis[3][3];
		double K[3];
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
