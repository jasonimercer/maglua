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

#include "mersennetwister.h"

int MTRand::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushfstring(L, "%s generates random variables using the Mersenne Twister. Mersenne Twister random number generator -- a C++ class MTRand. Based on code by Makoto Matsumoto, Takuji Nishimura, and Shawn Cokus. Richard J. Wagner  v1.1  28 September 2009  wagnerr@umich.edu", MTRand::slineage(0));
		lua_pushstring(L, ""); //input, empty
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
		
	return RNG::help(L);
}
