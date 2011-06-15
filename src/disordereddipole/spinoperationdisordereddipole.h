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

#ifndef SPINOPERATIONDIPOLEDISORDERED
#define SPINOPERATIONDIPOLEDISORDERED

#include "spinoperation.h"

using namespace std;

class DisorderedDipole : public SpinOperation
{
public:
	DisorderedDipole(int nx, int ny, int nz);
	virtual ~DisorderedDipole();
	
	bool apply(SpinSystem* ss);
	
	double g;

	void setPosition(int site, double px, double py, double pz);
	
	virtual void encode(buffer* b) const;
	virtual int  decode(buffer* b);

//private:
	double* posx;
	double* posy;
	double* posz;
};

void lua_pushDisorderedDipole(lua_State* L, DisorderedDipole* d);
DisorderedDipole* checkDisorderedDipole(lua_State* L, int idx);
void registerDisorderedDipole(lua_State* L);

#endif
