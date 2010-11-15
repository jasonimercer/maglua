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

#ifndef SPINOPERATIONAPPLIEDFIELD
#define SPINOPERATIONAPPLIEDFIELD

#include "spinoperation.h"

class AppliedField : public SpinOperation
{
public:
	AppliedField(int nx, int ny, int nz);
	virtual ~AppliedField();
	
	bool apply(SpinSystem* ss);

	double B[3];

	virtual void encode(buffer* b) const;
	virtual int  decode(buffer* b);
};

void lua_pushAppliedField(lua_State* L, AppliedField* ap);
AppliedField* checkAppliedField(lua_State* L, int idx);
void registerAppliedField(lua_State* L);


#endif
