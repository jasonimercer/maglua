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

#include "encodable.h"
#include <stdlib.h>
#include <string.h>

void ensureSize(int add, buffer* b)
{
	if(b->pos + add >= b->size)
	{
		b->size *= 2;
		if(b->pos + add >= b->size)
			b->size += add;

		b->buf = (char*)realloc(b->buf, b->size);
	}
}

void encodeBuffer(const void* s, const int len, buffer* b)
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


void decodeBuffer(void* dest, const int len, buffer* b)
{
	memcpy(dest, b->buf+b->pos, len);
	b->pos += len;
}
int decodeInteger(buffer* b)
{
	int i;
	decodeBuffer(&i, sizeof(int), b);
	return i;
}
double decodeDouble(buffer* b)
{
	double d;
	decodeBuffer(&d, sizeof(double), b);
	return d;
}

