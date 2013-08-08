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

#ifndef SPINOPERATIONAPPLIEDFIELD_HETERO
#define SPINOPERATIONAPPLIEDFIELD_HETERO

#include "spinoperation.h"
#include "array.h"


class AppliedField_Heterogeneous : public SpinOperation
{
public:
	AppliedField_Heterogeneous(int nx=32, int ny=32, int nz=1);
	virtual ~AppliedField_Heterogeneous();
	
	LINEAGE2("AppliedField.Heterogeneous", "SpinOperation")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);
	
	bool apply(SpinSystem* ss);
	bool apply(SpinSystem** ss, int n);

	dArray* hx;
	dArray* hy;
	dArray* hz;
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
};

#endif
