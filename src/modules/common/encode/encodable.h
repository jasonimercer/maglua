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



typedef struct buffer
{
	char* buf;
	int pos;
	int size;
}buffer;

#define ENCODE_UNKNOWN      0

#define ENCODE_SPINSYSTEM   1
#define ENCODE_ANISOTROPY   2
#define ENCODE_APPLIEDFIELD 3
#define ENCODE_DIPOLE       4
#define ENCODE_EXCHANGE     5
#define ENCODE_THERMAL      6

#define ENCODE_LLGCART      7
#define ENCODE_LLGQUAT      8
#define ENCODE_LLGFAKE      9
#define ENCODE_LLGALIGN    10

#define ENCODE_INTERP2D    11
#define ENCODE_INTERP1D    12

// This is a base class for classes that 
// can be encoded into and from a char stream,
class Encodable
{
public:
	Encodable(int t) : type(t) {};
	virtual ~Encodable() {};
	
	virtual void encode(buffer* b) = 0;
	virtual int  decode(buffer* b) = 0;
	
	int type;
	lua_State* L;
};






#ifdef WIN32
#ifdef ENCODE_EXPORTS
ENCODE_API   void encodeBuffer(const void* s, const int len, buffer* b);
ENCODE_API   void encodeDouble(const double d, buffer* b);
ENCODE_API   void encodeInteger(const int i, buffer* b);
ENCODE_API    int decodeInteger(buffer* b);
ENCODE_API double decodeDouble(buffer* b);
ENCODE_API   void decodeBuffer(void* dest, const int len, buffer* b);
#else
#include <windows.h>
inline   void encodeBuffer(const void* s, const int len, buffer* b)
{
	typedef void (*func)(const void*, const int, buffer*); 
	static func thefunc = 0;
	if(!thefunc)
		thefunc = (func) GetProcAddress(GetModuleHandle(NULL), "encodeBuffer");

	if(!thefunc)
	{
		printf("failed to load encodeBuffer\n");
		return;
	}
	thefunc(s,len,b);
}

inline void encodeDouble(const double d, buffer* b)
{
	typedef void (*func)(const double, buffer*); 
	static func thefunc = 0;
	if(!thefunc)
		thefunc = (func) GetProcAddress(GetModuleHandle(NULL), "encodeDouble");

	if(!thefunc)
	{
		printf("failed to load encodeDouble\n");
		return;
	}
	thefunc(d,b);
}

inline  void encodeInteger(const int i, buffer* b)
{
	typedef void (*func)(const int, buffer*); 
	static func thefunc = 0;
	if(!thefunc)
		thefunc = (func) GetProcAddress(GetModuleHandle(NULL), "encodeInteger");

	if(!thefunc)
	{
		printf("failed to load encodeInteger\n");
		return;
	}
	thefunc(i,b);
}

inline int decodeInteger(buffer* b)
{
	typedef int (*func)(buffer*);
	static func thefunc = 0;
	if(!thefunc)
		thefunc = (func) GetProcAddress(GetModuleHandle(NULL), "decodeInteger");

	if(!thefunc)
	{
		printf("failed to load decodeInteger\n");
		return 0;
	}
	return thefunc(b);
}

inline double decodeDouble(buffer* b)
{
	typedef double (*func)(buffer*);
	static func thefunc = 0;
	if(!thefunc)
		thefunc = (func) GetProcAddress(GetModuleHandle(NULL), "decodeDouble");

	if(!thefunc)
	{
		printf("failed to load decodeDouble\n");
		return 0;
	}
	return thefunc(b);
}

inline void decodeBuffer(void* dest, const int len, buffer* b)
{
	typedef void (*func)(void*, const int, buffer*); 
	static func thefunc = 0;
	if(!thefunc)
		thefunc = (func) GetProcAddress(GetModuleHandle(NULL), "decodeBuffer");

	if(!thefunc)
	{
		printf("failed to load decodeBuffer\n");
		return;
	}
	thefunc(dest,len,b);
}
#endif
#else
ENCODE_API   void encodeBuffer(const void* s, const int len, buffer* b);
ENCODE_API   void encodeDouble(const double d, buffer* b);
ENCODE_API   void encodeInteger(const int i, buffer* b);
ENCODE_API    int decodeInteger(buffer* b);
ENCODE_API double decodeDouble(buffer* b);
ENCODE_API   void decodeBuffer(void* dest, const int len, buffer* b);
#endif
#endif
