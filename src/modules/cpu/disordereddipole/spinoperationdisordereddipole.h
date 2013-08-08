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

#include "array.h"
#include "spinoperation.h"

using namespace std;

class DisorderedDipole : public SpinOperation
{
public:
	DisorderedDipole(int nx=1, int ny=1, int nz=1);
	virtual ~DisorderedDipole();
	
	LINEAGE2("DisorderedDipole", "SpinOperation")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);

	bool apply(SpinSystem* ss);
	double g;

	void setPosition(int site, double px, double py, double pz);
	
	void init();
	void deinit();
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);

//private:
	dArray* posx;
	dArray* posy;
	dArray* posz;
};

#endif
