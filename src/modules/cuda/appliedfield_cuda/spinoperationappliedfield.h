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

#ifndef SPINOPERATIONAPPLIEDFIELD
#define SPINOPERATIONAPPLIEDFIELD

#include "spinoperation.h"

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef APPLIEDFIELDCUDA_EXPORTS
  #define APPLIEDFIELDCUDA_API __declspec(dllexport)
 #else
  #define APPLIEDFIELDCUDA_API __declspec(dllimport)
 #endif
#else
 #define APPLIEDFIELDCUDA_API 
#endif

class APPLIEDFIELDCUDA_API AppliedField : public SpinOperation
{
public:
	AppliedField(int nx, int ny, int nz);
	virtual ~AppliedField();
	
	bool apply(SpinSystem* ss);

	double B[3];

	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
};

void lua_pushAppliedField(lua_State* L, Encodable* ap);
AppliedField* checkAppliedField(lua_State* L, int idx);
void registerAppliedField(lua_State* L);


#endif
