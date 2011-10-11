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

class RNG;
class LLG;
class Thermal : public SpinOperation
{
public:
	Thermal(int nx, int ny, int nz);
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
	
	virtual void encode(buffer* b) const;
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
void lua_pushThermal(lua_State* L, Thermal* th);


#endif
