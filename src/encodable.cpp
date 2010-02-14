#include "encodable.h"
#include <stdlib.h>
#include <string.h>

void ensureSize(int add, buffer* b)
{
	if(b->pos + add >= b->size)
	{
		if(b->size)
		{
			b->size *= 2;
			if(b->pos + add >= b->size)
				b->size += add;
		}
		else
			b->size += add;
		b->buf = (char*)realloc(b->buf, b->size);
	}
}

void encodeBuffer(const void* s, int len, buffer* b)
{
	ensureSize(len, b);
	memcpy(b->buf + b->pos, s, len);
	b->pos += len;
}

void encodeDouble(const double d, buffer* b)
{
	encodeBuffer(&d, sizeof(d), b);
}

void encodeInteger(const int i, buffer* b)
{
	encodeBuffer(&i, sizeof(i), b);
}


int decodeInteger(buffer* b)
{
	int i;
	memcpy(&i, b->buf+b->pos, sizeof(int));
	b->pos += sizeof(int);
	return i;
}
double decodeDouble(buffer* b)
{
	double d;
	memcpy(&d, b->buf+b->pos, sizeof(double));
	b->pos += sizeof(double);
	return d;
}
