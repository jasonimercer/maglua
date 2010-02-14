#ifndef ENCODABLE_H
#define ENCODABLE_H

typedef struct buffer
{
	char* buf;
	int pos;
	int size;
}buffer;

#define ENCODE_UNKNOWN    0
#define ENCODE_SPINSYSTEM 1

// This is a base class for classes that 
// can be encoded into and from a char stream,
class Encodable
{
public:
	Encodable(int t = ENCODE_UNKNOWN) : type(t) {};
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

#endif
