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

#ifndef SPINOPERATIONDIPOLEDISORDERED
#define SPINOPERATIONDIPOLEDISORDERED

#include "spinoperation.h"

using namespace std;

class DipoleDisordered : public SpinOperation
{
public:
	DipoleDisordered(int nx, int ny, int nz);
	virtual ~DipoleDisordered();
	
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

void lua_pushDipoleDisordered(lua_State* L, DipoleDisordered* d);
DipoleDisordered* checkDipoleDisordered(lua_State* L, int idx);
void registerDipoleDisordered(lua_State* L);


#endif
