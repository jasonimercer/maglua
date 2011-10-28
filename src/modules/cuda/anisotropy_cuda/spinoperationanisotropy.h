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

 #ifdef ANISOTROPYCUDA_EXPORTS
  #define ANISOTROPYCUDA_API __declspec(dllexport)
 #else
  #define ANISOTROPYCUDA_API __declspec(dllimport)
 #endif
#else
 #define ANISOTROPYCUDA_API 
#endif

class ANISOTROPYCUDA_API Anisotropy : public SpinOperation
{
public:
	Anisotropy(int nx=32, int ny=32, int nz=1);
	virtual ~Anisotropy();
	
	bool apply(SpinSystem* ss);
	void addAnisotropy(int site, double nx, double ny, double nz, double K);
	
	double* d_nx;
	double* d_ny;
	double* d_nz;
	double* d_k;
	
	double* h_nx;
	double* h_ny;
	double* h_nz;
	double* h_k;
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
	
	void sync_dh(bool force=false);
	void sync_hd(bool force=false);

	bool new_host; // if host data is most recent
	bool new_device;

	void init();
	void deinit();
};

ANISOTROPYCUDA_API void lua_pushAnisotropy(lua_State* L, Encodable* _ani);
ANISOTROPYCUDA_API Anisotropy* checkAnisotropy(lua_State* L, int idx);
ANISOTROPYCUDA_API void registerAnisotropy(lua_State* L);


#endif
