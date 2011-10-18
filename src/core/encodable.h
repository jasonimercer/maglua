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
#include "luacommon.h"

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
class CORE_API Encodable
{
public:
	Encodable(int t) : type(t) {};
	virtual ~Encodable() {};
	
	virtual void encode(buffer* b) const = 0;
	virtual int  decode(buffer* b) = 0;
	
	int type;
};

  void encodeBuffer(const void* s, const int len, buffer* b);
  void encodeDouble(const double d, buffer* b);
  void encodeInteger(const int i, buffer* b);
   int decodeInteger(buffer* b);
double decodeDouble(buffer* b);
  void decodeBuffer(void* dest, const int len, buffer* b);

#endif
