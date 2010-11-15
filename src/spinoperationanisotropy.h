/******************************************************************************
* Copyright (C) 2008-2010 Jason Mercer.  All rights reserved.
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

class Anisotropy : public SpinOperation
{
public:
	Anisotropy(int nx, int ny, int nz);
	virtual ~Anisotropy();
	
	bool apply(SpinSystem* ss);

	double* ax;
	double* ay;
	double* az;
	double* strength;
	
	virtual void encode(buffer* b) const;
	virtual int  decode(buffer* b);

	void init();
	void deinit();
};

void lua_pushAnisotropy(lua_State* L, Anisotropy* ani);
Anisotropy* checkAnisotropy(lua_State* L, int idx);
void registerAnisotropy(lua_State* L);


#endif
