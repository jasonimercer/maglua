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

#ifndef SPINOPERATIONEXCHANGE
#define SPINOPERATIONEXCHANGE

#include "spinoperation.h"

class CORE_API Exchange : public SpinOperation
{
public:
	Exchange(int nx, int ny, int nz);
	virtual ~Exchange();
	
	bool apply(SpinSystem* ss);

	void addPath(int site1, int site2, double strength);

	virtual void encode(buffer* b) const;
	virtual int  decode(buffer* b);
	void opt();

	typedef struct sss
	{
		int fromsite;
		int tosite;
		double strength;
	} sss;

private:
	void deinit();
	
	int size;
	int num;
	sss* pathways;
	
// 	int* fromsite;
// 	int* tosite;
// 	double* strength;
};

CORE_API Exchange* checkExchange(lua_State* L, int idx);
CORE_API void registerExchange(lua_State* L);
CORE_API void lua_pushExchange(lua_State* L, Exchange* ex);

#endif
