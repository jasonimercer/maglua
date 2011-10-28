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

#ifndef SPINOPERATIONTHERMAL
#define SPINOPERATIONTHERMAL

#include "spinoperation.h"
#include "hybridtaus.hpp"
#include <stdint.h>

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef THERMALCUDA_EXPORTS
  #define THERMALCUDA_API __declspec(dllexport)
 #else
  #define THERMALCUDA_API __declspec(dllimport)
 #endif
#else
 #define THERMALCUDA_API 
#endif

class RNG;
class LLG;
class Thermal : public SpinOperation
{
public:
	Thermal(int nx=32, int ny=32, int nz=1);
	virtual ~Thermal();
	
	bool apply(RNG* rand, SpinSystem* ss);
	bool apply(SpinSystem* ss) {return false;};

	void scaleSite(int px, int py, int pz, double strength);

	void init();
	void deinit();
	
	double temperature;


	state_t* d_state;
	float* d_rngs;
	
	void sync_dh(bool force=false);
	void sync_hd(bool force=false);
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
	
	bool new_device;
	bool new_host;
private:
	double* d_scale;
	double* h_scale;

	int twiddle;
};

Thermal* checkThermal(lua_State* L, int idx);
void registerThermal(lua_State* L);
void lua_pushThermal(lua_State* L, Encodable* th);


#endif
