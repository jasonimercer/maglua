/******************************************************************************
* Copyright (C) 2008-2014 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#include "checkpointer.h"
#include <stdlib.h>
#include <luamigrate.h>
#include <zlib.h>
#include "base64.h"

#define OP_UNKNOWN -2
#define OP_NONE -1
#define OP_PACK 0
#define OP_ZLIB 1
#define OP_B64  2

#define HEADER_SIZE (sizeof(int)*4 + 4)

static void decodeHeader(const char* buffer, char* c4, int& new_encoding, int& old_encoding, int& new_size, int& old_size);

static int sure_fwrite(const void* _data, int sizeelement, int numelement, FILE* f)
{
    int sz = 0;
    const char* data = (const char*)_data;

    do
    {
        int w = fwrite(data + sz*sizeelement, sizeelement, numelement - sz, f);
        sz += w;

        if(w == 0)
        {
            return 0;
        }
    }while(sz < numelement);

    return sz;
}


static int sure_fread(void* _data, int sizeelement, int numelement, FILE* f)
{
    char* data = (char*)_data;
    int sz = 0;

    do
    {
        int r = fread(data + sz*sizeelement, sizeelement, numelement - sz, f);
        sz += r;

        if(r == 0)
        {
            return 0;
        }
    }while(sz < numelement);

    return sz;
}



Checkpointer::Checkpointer()
	: LuaBaseObject(hash32(Checkpointer::typeName()))
{
    b = 0;//(char*)malloc(32);
	clear();
	checksum = 0;
}



int Checkpointer::luaInit(lua_State* L)
{
	for(int i=1; i<=lua_gettop(L); i++)
	{
		l_add(L, i);
	}

	return 0;
}

Checkpointer::~Checkpointer()
{
	clear();
	setData(0);
	delete b;
}


void Checkpointer::clear()
{
	if(b)
	{
		if(b->buf)
			free(b->buf);
		if(b->debug)
			fclose(b->debug);
		b->debug = 0;
	}
	b = new buffer;
    b->size = 32;
    b->pos  = 0;
	b->buf = 0;

	buffer_unref(L, b);

	b->encoded.clear();
	b->encoded_table_refs.clear();
	b->encoded_table_pointers.clear();
	
	n = 0;
	end_pos = 0;
	has_checksum = false;
	setData((char*)malloc(b->size));
	for(int i=0; i<b->size; i++)
		b->buf[i] = 0;
}

int Checkpointer::l_tostring(lua_State* L)
{
	int s = currentState();

	switch(s)
	{
	case OP_B64:
		lua_pushstring(L, b->buf);
		return 1;
	}

	return luaL_error(L, "Internal state cannot be converted to a string");
}


int Checkpointer::l_fromstring(lua_State* L, int idx)
{
	clear();

	const char* s = lua_tostring(L, idx);
	const int len = strlen(s);
	if(len > 6)
	{
		char qq[5];
		memcpy(qq, s, 3);
		qq[3] = 0;

		if(strncmp(qq, "QjY", 3) == 0)
		{
			char* q = (char*)malloc(len+1);

			memcpy(q, s, len+1);
			q[len] = 0;
			setData(q);
			
			return 0;
		}
		return luaL_error(L, "Data stream not recognized");
	}

	return luaL_error(L, "Data stream too short to identify");
}


int Checkpointer::l_add(lua_State* L, int idx)
{
	if(currentState() == OP_NONE)
	{
		if(debug_file.length())
		{
			if(b->debug == 0)
			{
				b->debug = fopen(debug_file.c_str(), "w");
			}
		}
		_exportLuaVariable(L, idx, b);
		has_checksum = false;
		n++;
		return 0;
	}
	return luaL_error(L, "Cannot add data to Checkpointer with internal encoding");
}


int Checkpointer::l_get(lua_State* L)
{
	if(currentState() == OP_NONE)
	{
		buffer b2;
		b2.pos = 0;
		b2.size = b->size;
		b2.buf = b->buf;
		for(int i=0; i<n; i++)
			_importLuaVariable(L, &b2);

		b2.buf = 0;
		buffer_unref(L, &b2);

		return n;
	}
	return luaL_error(L, "Cannot get data from Checkpointer with internal encoding");
}

int Checkpointer::l_gettable(lua_State* L)
{
	if(currentState() == OP_NONE)
	{
		lua_newtable(L);
		buffer b2;
		b2.pos = 0;
		b2.size = b->size;
		b2.buf = b->buf;
		for(int i=0; i<n; i++)
		{
			lua_pushinteger(L, i+1);
			_importLuaVariable(L, &b2);
			lua_settable(L, -3);
		}

        buffer_unref(L, &b2);
		b2.buf = 0;
		return 1;
	}
	return luaL_error(L, "Cannot get data from Checkpointer with internal encoding");
}

int Checkpointer::l_checksum(lua_State* L)
{
	calculateChecksum();
	lua_pushinteger(L, checksum);
	return 1;
}

int Checkpointer::l_copy(lua_State* L)
{
	Checkpointer* c = new Checkpointer;

	if(b)
	{
		c->b->size = b->size;
		c->b->pos = b->pos;
		int iss = internalStateSize();
		
		c->b->buf = (char*)malloc(iss);
		memcpy(c->b->buf, b->buf, iss);

		for(unsigned int i=0; i< b->encoded_table_refs.size(); i++)
		{
			c->b->encoded_table_refs.push_back( b->encoded_table_refs[i] );
		}
		for(unsigned int i=0; i< b->encoded_table_pointers.size(); i++)
		{
			c->b->encoded_table_pointers.push_back( b->encoded_table_pointers[i] );
		}

		
		c->n = n;
	}

	luaT_push<Checkpointer>(L, c);

	return 1;
}


void Checkpointer::encode(buffer* b)
{
	ENCODE_PREAMBLE  
}

int  Checkpointer::decode(buffer* b)
{
	return 0;
}



int Checkpointer::deoperate_data(lua_State* L)
{
	char sig[4];
	int new_encoding;
	int old_encoding;
	int new_size;
	int old_size;

#if 0
	{
		const char* d = data();
		int i;
		memcpy(&i, d+HEADER_SIZE, sizeof(int));
		fprintf(stderr, "!!!!    %c%c%c%c   %i\n", d[0], d[1], d[2], d[3], i);
	}
#endif

	decodeHeader(data(), sig, new_encoding, old_encoding, new_size, old_size);

	// special case where n is after the header
	if(old_encoding == OP_NONE && new_encoding == OP_PACK)
	{
		char* d = (char*)malloc(old_size+HEADER_SIZE+sizeof(int));
		memcpy(&n, data()+HEADER_SIZE, sizeof(int));
		memcpy(d, data()+HEADER_SIZE+sizeof(int), old_size);

		setData(d);
		b->size = old_size;
		b->pos = old_size;
		return 0;
	}

	switch(new_encoding)
	{
	case OP_PACK: // unpack from file format
	{
		char* d = (char*)malloc(old_size);
		memcpy(d, data()+HEADER_SIZE, old_size);
		setData(d);
        break;
	}
	case OP_ZLIB: //compressed
	{
		char* d = (char*)malloc(old_size);

		unsigned long compressed_size = new_size;
		unsigned long inflated_size = old_size;

		// printf("src size: %li bytes\n", compressed_size);
		// printf("made a buffer for inflation: %li bytes\n", inflated_size);

		// int uncompress(Bytef * dest, uLongf * destLen, const Bytef * source, uLong sourceLen);

		int retval = uncompress((unsigned char*)d, &inflated_size, (unsigned char*)data() + HEADER_SIZE, compressed_size); 

		if(retval != Z_OK)
		{
			return luaL_error(L, "Error in compressed datastream.");
		}

		// printf("retval = %i, OK = %i, Z_BUF_ERR=%i, Z_MEM_ERR=%i, Z_DATA_ERROR=%i\n", retval, Z_OK, Z_BUF_ERROR, Z_MEM_ERROR, Z_DATA_ERROR);

		setData(d);
		break;
	}
	case OP_B64: // unpack from b64 encoding
	{
		std::string dataE( data() );
		std::string dataD = base64_decode(dataE);

		char* e = (char*)malloc(old_size);
		memcpy(e, dataD.data()+HEADER_SIZE, old_size);

		setData(e);
		break;
	}
	}

	return 0;
}

// void compress_data(int direction=1); // operation 1
// void uuencode_data(int direction=1); // operation 2
// void b64encode_data(int direction=1); // operation 3

static char* makeBufferHeader(const char* c4, int new_encoding, int old_encoding, int new_size, int old_size, char* b=0)
{
	if(!b)
	    b = (char*)malloc( new_size );
	
	int d[4];
	d[0] = new_encoding;
	d[1] = old_encoding;
	d[2] = new_size;
	d[3] = old_size;

	memcpy(b, c4, 4);
	memcpy(b+4, d, sizeof(int)*4);

	return b;
}

static void decodeHeader(const char* buffer, char* c4, int& new_encoding, int& old_encoding, int& new_size, int& old_size)
{
	int op = Checkpointer::currentStateS(buffer);
	int hsz = HEADER_SIZE;
	c4[0] = 0;
	if(op == OP_B64)
	{
		char* dd = (char*)malloc(hsz);
		std::string ss = base64_decode(buffer, hsz*2);
		memcpy(dd, ss.data(), hsz);

		memcpy(c4, dd, 4);
		memcpy(&new_encoding, dd+4+sizeof(int)*0, sizeof(int));
		memcpy(&old_encoding, dd+4+sizeof(int)*1, sizeof(int));
		memcpy(&new_size,     dd+4+sizeof(int)*2, sizeof(int));
		memcpy(&old_size,     dd+4+sizeof(int)*3, sizeof(int));

		free(dd);
		return;
	}

	if(op == OP_NONE || op == OP_UNKNOWN)
	{
		new_encoding = op;
		old_encoding = op;
		new_size = 0;
		old_size = 0;
		return;
	}

	if(op == OP_PACK || op == OP_ZLIB)
	{
		const char* dd = buffer;
		memcpy(c4, dd, 4);
		memcpy(&new_encoding, dd+4+sizeof(int)*0, sizeof(int));
		memcpy(&old_encoding, dd+4+sizeof(int)*1, sizeof(int));
		memcpy(&new_size,     dd+4+sizeof(int)*2, sizeof(int));
		memcpy(&old_size,     dd+4+sizeof(int)*3, sizeof(int));
		return;
	}

	return;
}



char* Checkpointer::data()
{
	return b->buf;
}

char* Checkpointer::setData(char* c)
{
	if(c != b->buf)
	{
		free(b->buf);
		b->buf = c;
	}
	return c;
}

int  Checkpointer::operate_data(lua_State* L, int idx)
{
	int op = lua_tointeger(L, idx) - 1;

	int iss = internalStateSize();

	char* new_buf = 0;

	switch(op)
	{
	case OP_PACK: // pack for file
	{
		int old_size = internalStateSize();
		int new_size = HEADER_SIZE + old_size;

		int new_state = OP_PACK;
		int old_state = currentState();
		int data_pos = HEADER_SIZE;
		if(old_state == OP_NONE)
		{
			new_size += sizeof(int); // to hold n
		}

		new_buf = makeBufferHeader("LUA", OP_PACK, currentState(), new_size, old_size);

		if(old_state == OP_NONE) // need to encode n
		{
			memcpy(new_buf + data_pos, &n, sizeof(int));
			data_pos += sizeof(int);
		}

		memcpy(new_buf+data_pos, data(), old_size);
		break;
	}
	case OP_ZLIB: // compress
	{
		if(currentState() != OP_PACK)
		{
			// fprintf(stderr, "current state = %i\n", currentState(-1));
			return luaL_error(L, "Must be in the Packed state to compress");
		}


		int data_pos = HEADER_SIZE;

		int old_size = internalStateSize();
		int new_size = data_pos + compressBound(old_size)+1;;

		int new_state = OP_ZLIB;
		int old_state = OP_PACK;

		new_buf = makeBufferHeader("ZZZ", new_state, old_state, new_size, old_size);

		uLongf destLen = new_size - data_pos;
		int retval = compress2((unsigned char*)new_buf + data_pos, &destLen, (const unsigned char*)data(), old_size, 9);

		// now we have a real dest len, let's patch it in
		new_size = data_pos + (int)destLen;

		makeBufferHeader("ZZZ", new_state, old_state, new_size, old_size, new_buf);
		break;
	}
	case OP_B64:
	{
		int data_pos = HEADER_SIZE;

		int old_size = internalStateSize();
		int new_size = old_size * 1.4 + 8; // 8 for good measure

		int new_state = OP_B64;
		int old_state = currentState();

		char* new_buf2 = makeBufferHeader("B64", new_state, old_state, new_size, old_size);

		memcpy(new_buf2 + data_pos, data(), old_size);

		std::string d = base64_encode((const unsigned char*)new_buf2, old_size + data_pos);

		// now we know how big the encoded string is so we can update the new_size
		// and get a good header
		new_size = d.size(); // data_pos + d.size() + 1;

		char* good_header = (char*) malloc(data_pos);
		makeBufferHeader("B64", new_state, old_state, new_size, old_size, good_header);
		std::string good_header_encode = base64_encode((const unsigned char*)good_header, data_pos);

		new_buf = (char*)malloc(d.size()+1);
		memcpy(new_buf, d.data(), d.size());
		new_buf[d.size()] = 0;
		
		const char* x = good_header_encode.data();
		for(int i=2; i<good_header_encode.size() && x[i] != '='; i++)
		{
			const int j = i-2;
			new_buf[j] = x[j];
		}

		free(new_buf2);
		free(good_header);

		break;
	}
	}


	if(new_buf)
		setData(new_buf);

	return 0;
}

int Checkpointer::currentStateS(const char* d)
{
	if(!d)
		return OP_UNKNOWN;

	char sig[4];
	memcpy(sig, d, 3);
	sig[3] = 0;

	if(strncmp(sig, "LUA", 3) == 0)
		return OP_PACK;

	if(strncmp(sig, "ZZZ", 3) == 0)
		return OP_ZLIB;

	if(strncmp(sig, "QjY", 3) == 0)
		return OP_B64;

	return OP_NONE;

}

int Checkpointer::currentState(const char* d)
{
	if(!d)
		return Checkpointer::currentStateS(data());
	return Checkpointer::currentStateS(d);
}



void Checkpointer::calculateChecksum()
{
	if(!has_checksum)
	{
		checksum = adler32(0L, Z_NULL, 0);
		if(b->buf)
			checksum = adler32(checksum, (const unsigned char*)b->buf, b->pos);
		has_checksum = true;
	}
}


int Checkpointer::internalStateSize()
{
	int op = currentState();

	if(op == OP_NONE)
	{
		if(b)
			return b->pos;
		return 0;
	}

	char sig[4];
	int new_encoding;
	int old_encoding;
	int new_size;
	int old_size;

	decodeHeader(data(), sig, new_encoding, old_encoding, new_size, old_size);
	return new_size;
}


int Checkpointer::l_savetofile(lua_State* L)
{
	int op = currentState();
	if(op < 0)
	{
		return luaL_error(L, "Internal state must be in encode state");
	}

	const char* fn = lua_tostring(L, 2);

	FILE* f = fopen(fn, "w");

	if(!f)
		return luaL_error(L, "failed to open file for writing");


	char sig[4];
    int new_encoding;
    int old_encoding;
    int new_size;
    int old_size;

    decodeHeader(data(), sig, new_encoding, old_encoding, new_size, old_size);

	sure_fwrite(data(), new_size, 1, f);
	fclose(f);

	return 0;
}

int Checkpointer::l_loadfromfile(lua_State* L)
{
	const char* fn = lua_tostring(L, 2);

	FILE* f = fopen(fn, "r");

	if(!f)
		return luaL_error(L, "failed to open file for reading");

	const int chunk = 1024*10;
	int size = chunk;

	char* d = (char*)malloc(size);
	int pos = 0;
	

	while(fread(d + pos, chunk, 1, f) == 1)
	{
		size += chunk;
		pos += chunk;
		d = (char*)realloc(d, size);
	}

	setData(d);

	return 0;
}



static int _l_tostring(lua_State* L)
{
	LUA_PREAMBLE(Checkpointer, c, 1);
	return c->l_tostring(L);
}

static int _l_fromstring(lua_State* L)
{
	LUA_PREAMBLE(Checkpointer, c, 1);
	return c->l_fromstring(L, 2);
}


static int _l_add(lua_State* L)
{
	LUA_PREAMBLE(Checkpointer, c, 1);
	for(int i=2; i<=lua_gettop(L); i++)
	{
		c->l_add(L, i);
	}
	return 0;
}

static int _l_get(lua_State* L)
{
	LUA_PREAMBLE(Checkpointer, c, 1);
	return c->l_get(L);
}

static int _l_checksum(lua_State* L)
{
	LUA_PREAMBLE(Checkpointer, c, 1);
	return c->l_checksum(L);
}

static int _l_savetofile(lua_State* L)
{
	LUA_PREAMBLE(Checkpointer, c, 1);
	return c->l_savetofile(L);
}

static int _l_loadfromfile(lua_State* L)
{
	LUA_PREAMBLE(Checkpointer, c, 1);
	return c->l_loadfromfile(L);
}


static int _l_operate(lua_State* L)
{
	LUA_PREAMBLE(Checkpointer, c, 1);
	for(int i=2; i<=lua_gettop(L); i++)
	{
		c->operate_data(L, i);	
	}
	return 0;
}

static int _l_deoperate(lua_State* L)
{
	LUA_PREAMBLE(Checkpointer, c, 1);
	c->deoperate_data(L);	
	return 0;
}

static int _l_gettable(lua_State* L)
{
	LUA_PREAMBLE(Checkpointer, c, 1);
	return c->l_gettable(L);	
}

static int _l_transformation(lua_State* L)
{
	LUA_PREAMBLE(Checkpointer, c, 1);

	lua_pushinteger(L, c->currentState()+1);

	return 1;
}

static int _l_decodeheader(lua_State* L)
{
	LUA_PREAMBLE(Checkpointer, c, 1);

	char sig[4];
	int new_encoding;
	int old_encoding;
	int new_size;
	int old_size;

	decodeHeader(c->data(), sig, new_encoding, old_encoding, new_size, old_size);

	lua_pushstring(L, sig);
	lua_pushinteger(L, new_encoding+1);
	lua_pushinteger(L, old_encoding+1);
	lua_pushinteger(L, new_size);
	lua_pushinteger(L, old_size);
	return 5;
}

static int l_stateSize(lua_State* L)
{
	LUA_PREAMBLE(Checkpointer, c, 1);

	buffer* b = c->b;

	lua_pushinteger(L, c->internalStateSize());
	return 1;
}

static int _l_copy(lua_State* L)
{
    LUA_PREAMBLE(Checkpointer, c, 1);
	return c->l_copy(L);
}

static int _l_setdbfile(lua_State* L)
{
    LUA_PREAMBLE(Checkpointer, c, 1);
	if(lua_isstring(L, 2))
		c->debug_file = lua_tostring(L, 2);
	else
		c->debug_file = "";
	return 0;
}

int Checkpointer::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		char txt[1024];
		
		snprintf(txt, 1024, "Checkpointing Object. This can be used to encode data into various formats. "
			"You can also build custom checkpoint functions. Consider the following replacement for the "
		    "default checkpointToFile which will compress the data:"
			"<pre>function checkpointToFile(fn, ...)\n"
			"    local cp = Checkpointer.new()\n"
			"    cp:addTable(arg)\n"
            "    cp:transform(\"Compress\")\n"
			"    cp:saveToFile(fn)\n"
			"end</pre>It should be noted that the default checkpointFromFile does not need to be changed to deal with the new files.");



		lua_pushstring(L, txt);
		lua_pushstring(L, "0 or more data: These data will be added to the encoded state.");
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
	
	if(func == _l_add)
	{
		lua_pushstring(L, "Add data to the internal encoded state.");
		lua_pushstring(L, "0 or more data: These data will be added to the encoded state.");
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	if(func == _l_get)
	{
		lua_pushstring(L, "Get all data encoded in the internal state. Note: there is no way to decode only parts of the internal state.");
		lua_pushstring(L, "");
		lua_pushstring(L, "0 or more data: The encoded data.");
		return 3;
	}
	
	if(func == _l_checksum)
	{
		lua_pushstring(L, "Calculate the Adler-32 checksum for the encoded data.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: Alder-32 checksum.");
		return 3;
	}
	
	if(func == _l_savetofile)
	{
		lua_pushstring(L, "Save the internal state to a file");
		lua_pushstring(L, "1 String: The string is the filename.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == _l_loadfromfile)
	{
		lua_pushstring(L, "Load the internal state from a file");
		lua_pushstring(L, "1 String: The string is the filename which contains the state.");
		lua_pushstring(L, "");
		return 3;
	}

	return LuaBaseObject::help(L);
}


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Checkpointer::luaMethods()
{
	if(m[127].name)return m;

	static const luaL_Reg _m[] =
	{
		{"add", _l_add},
		{"get", _l_get},
		{"checksum", _l_checksum},
		{"saveToFile", _l_savetofile},
		{"loadFromFile", _l_loadfromfile},
		{"stateSize", l_stateSize},
		{"toString", _l_tostring},
		{"fromString", _l_fromstring},
		{"copy", _l_copy},

		{"_operate", _l_operate},
		{"_deoperate", _l_deoperate},
		{"_transformation", _l_transformation},
		{"_decodeHeader", _l_decodeheader},
		{"_getTable", _l_gettable},

		{"_setDebugFile", _l_setdbfile},

		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}





static int l_getmetatable(lua_State* L)
{
	if(!lua_isstring(L, 1))
		return luaL_error(L, "First argument must be a metatable name");
	luaL_getmetatable(L, lua_tostring(L, 1));
	return 1;
}

#include "checkpointer_luafuncs.h"
int  checkpointer_register(lua_State* L)
{
	luaT_register<Checkpointer>(L);
	
 	lua_pushcfunction(L, l_getmetatable);
	lua_setglobal(L, "maglua_getmetatable");

	if(luaL_dostringn(L, __checkpointer_luafuncs(), __checkpointer_luafuncs_name()))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		return luaL_error(L, lua_tostring(L, -1));
	}
	
	lua_pushnil(L);
	lua_setglobal(L, "maglua_getmetatable");
}



