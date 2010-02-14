#ifndef ENCODABLE_H
#define ENCODABLE_H

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

// This is a base class for classes that 
// can be encoded into and from a char stream,
class Encodable
{
public:
	Encodable(int t) : type(t) {};
	virtual ~Encodable() {};
	
	virtual void encode(buffer* b) const = 0;
	virtual int  decode(buffer* b) = 0;
	
	int type;
};

  void encodeBuffer(const void* s, int len, buffer* b);
  void encodeDouble(const double d, buffer* b);
  void encodeInteger(const int i, buffer* b);
   int decodeInteger(buffer* b);
double decodeDouble(buffer* b);
  void decodeBuffer(void* dest, int len, buffer* b);

#endif
