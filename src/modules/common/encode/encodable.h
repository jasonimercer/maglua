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

#if 0

#ifndef ENCODABLE_H
#define ENCODABLE_H
#include "factory.h"


#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef ENCODE_EXPORTS
  #define ENCODE_API __declspec(dllexport)
 #else
  #define ENCODE_API __declspec(dllimport)
 #endif
#else
 #define ENCODE_API 
#endif


ENCODE_API typedef struct buffer
{
	char* buf;
	int pos;
	int size;
}buffer;

#define ENCODE_UNKNOWN      0

#define ENCODE_SPINSYSTEM   hash32("SpinSystem")
#define ENCODE_ANISOTROPY   hash32("Anisotropy")
#define ENCODE_APPLIEDFIELD hash32("AppliedField")
#define ENCODE_DIPOLE       hash32("Dipole")
#define ENCODE_EXCHANGE     hash32("Exchange")
#define ENCODE_THERMAL      hash32("Thermal")

#define ENCODE_LLGCART      hash32("LLGCart")
#define ENCODE_LLGQUAT      hash32("LLGQuat")
#define ENCODE_LLGFAKE      hash32("LLGFake")
#define ENCODE_LLGALIGN     hash32("LLGALign")

#define ENCODE_INTERP2D    hash32("Interpolate2D")
#define ENCODE_INTERP1D    hash32("interpolate1D")
#define ENCODE_MAGNETOSTATIC hash32("Magnetostatic")

#define ENCODE_SHORTRANGE  hash32("ShortRange")

// This is a base class for classes that 
// can be encoded into and from a char stream,
class ENCODE_API Encodable
{
public:
	Encodable(int t) : type(t) {};
	virtual ~Encodable() {};
	
	virtual void encode(buffer* b) = 0;
	virtual int  decode(buffer* b) = 0;
	
	int type;
	lua_State* L;
};

extern "C"
{
ENCODE_API   void encodeBuffer(const void* s, const int len, buffer* b);
ENCODE_API   void encodeDouble(const double d, buffer* b);
ENCODE_API   void encodeInteger(const int i, buffer* b);
ENCODE_API    int decodeInteger(buffer* b);
ENCODE_API double decodeDouble(buffer* b);
ENCODE_API   void decodeBuffer(void* dest, const int len, buffer* b);
}
#endif

#endif