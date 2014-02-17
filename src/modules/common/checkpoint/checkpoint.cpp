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

// 
// CheckPoint adds 2 functions.
// 
// checkpointSave(filename, a, b, c, d, ...)
//  saves the values a, b, c, d ... to the file: filename
// 
// a, b, c, d, ... = checkpointLoad(fn)
//  loads the variables in the file: filename to the 
//  variables a, b, c, d, ...

#include "luamigrate.h"
#include "checkpoint.h"
#include <stdio.h>
#include <stdlib.h>

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

static char* bin_fmt(lua_State* L, const int start, const int end, int& buffer_size)
{
	const int n = end - start + 1;
	
	int bb_size = 1024 + sizeof(int);
	int bb_pos = 0;
	char* bigbuf = (char*)malloc(bb_size);

	char header[128];
	for(int i=0; i<128; i++)
		header[i] = 0;
	
	snprintf(header, 128, "CHECKPOINT");
	
	memcpy(bigbuf + bb_pos, header, 128);
	bb_pos += 128;
	
	memcpy(bigbuf + bb_pos, &n, sizeof(int));
	bb_pos += sizeof(int);
	
	for(int i=start; i<=end; i++)
	{
		int size;
		char* buf = exportLuaVariable(L, i, &size);
		
		// note: adding some extra memory below so we can get multiples of 3
		// in the uuencode case
		if(bb_pos + (int)sizeof(int) + size + 3 > bb_size)
		{
			bigbuf = (char*)realloc(bigbuf, bb_size + sizeof(int) + size + 3);
			bb_size = bb_size + sizeof(int) + size;
		}
		memcpy(bigbuf + bb_pos, &size, sizeof(int));
		bb_pos += sizeof(int);
		
		memcpy(bigbuf + bb_pos, buf, size);
		bb_pos += size;
		
		free(buf);
	}
	
	buffer_size = bb_pos;
	return bigbuf;	
}

const int b0011_1111 = 0x3F;
const int b0011_0000 = 0x30;
const int b0000_0011 = 0x03;
const int b0011_1100 = 0x3C;
const int b0000_1111 = 0x0F;
const int b0100_0000 = 0x40;
const int b1111_0000 = 0xF0;
const int b1100_0000 = 0xC0;

static void uuencode_chunk(const char* input3, char* output4)
{
	output4[0] = ((input3[0] >> 2) & b0011_1111); //6 upper bits
	output4[1] = ((input3[0] << 4) & b0011_0000) | ((input3[1] >> 4) & b0000_1111);
	output4[2] = ((input3[1] << 2) & b0011_1100) | ((input3[2] >> 6) & b0000_0011);
	output4[3] =   input3[2] & b0011_1111;
	
	for(int i=0; i<4; i++)
	{
		if(!output4[i])
		{
			output4[i] |= b0100_0000;
		}
		
		output4[i] += ' ';
	}
}

// return 1 for more, 0 for done
static int uuencode_line(const char* in_data, int& in_pos, const int in_size, char* out_data, int& out_pos)
{
	int retval = 0;
	int bytes_to_write = in_size - in_pos;
	if(bytes_to_write > 45)
	{
		bytes_to_write = 45;
		retval = 1; //done
	}

	
	out_data[out_pos] = bytes_to_write + 32;
	out_pos++;

	for(int i=0; i<bytes_to_write; i+=3)
	{
		uuencode_chunk(in_data+in_pos, out_data+out_pos);
		out_pos += 4;
		 in_pos += 3;
	}

	out_data[out_pos] = '\n';
	out_pos++;
	out_data[out_pos] = 0;
	//out_pos++; not incrementing here so we can write over eol

	return retval;
}

static char* uuencode(char* input, int size)
{
	char* buf = (char*)malloc(size*2+128); //making sure out buffer is more than big enough

	sprintf(buf, "begin 600 checkpoint.dat\n");
	
	// the buffer actually has some safety margins so we can stomp around like this
	input[size  ] = 0;
	input[size+1] = 0;
	input[size+2] = 0;

	int in_pos = 0;
	int out_pos = 25;

	while(uuencode_line(input, in_pos, size, buf, out_pos));

	sprintf(buf+out_pos, "%c\nend\n", 0x20 | 0x40);

	return buf;
}

static int l_checkpoint_save_to_string(lua_State* L)
{
	int size = 0;
	char* bigbuf = bin_fmt(L, 1, lua_gettop(L), size);

	char* uu_buf = uuencode(bigbuf, size);
	free(bigbuf);

	lua_pushstring(L, uu_buf);
	free(uu_buf);
	return 1;
}

static int l_checkpoint_save(lua_State* L)
{
	if(!lua_isstring(L, 1))
	{
		return luaL_error(L, "checkpointSave must have a filename as the first argument");
	}
	
	const char* fn = lua_tostring(L, 1);
	
	FILE* f = fopen(fn, "w");
	if(!f)
	{
		return luaL_error(L, "failed to open `%s' for writing", fn);
	}


	int bsz;
	char* buf = bin_fmt(L, 2, lua_gettop(L), bsz);
	
	if(!sure_fwrite(buf, bsz, 1, f))
	{
		free(buf);
		fclose(f);
		return luaL_error(L, "failed in write\n");
	}
	free(buf);
	fclose(f);
	return 0;
	
// 	if(!f)
// 	{
// 		return luaL_error(L, "failed to open `%s' for writing", fn);
// 	}
// 	
// 	char header[128];
// 	for(int i=0; i<128; i++)
// 		header[i] = 0;
// 	
// 	snprintf(header, 128, "CHECKPOINT");
// 	sure_fwrite(header, 1, 128, f); //write header
// 	sure_fwrite(&n, sizeof(int), 1, f); //write number of variables
// 	
// 	for(int i=2; i<=n+1; i++)
// 	{
// 		int size;
// 		char* buf = exportLuaVariable(L, i, &size);
// 		int sz = 0;
// 		
// 		if(!sure_fwrite(&size, sizeof(int), 1, f))
// 		{
// 			fclose(f);
// 			return luaL_error(L, "failed in write\n");
// 		}
// 		if(!sure_fwrite(buf, 1, size, f))
// 		{
// 			fclose(f);
// 			return luaL_error(L, "failed in write\n");
// 		}
// 		
// 		free(buf);
// 	}
// 	
// 	fclose(f);
// 	return 0;
}

static int l_checkpoint_load(lua_State* L)
{
	if(!lua_isstring(L, 1))
	{
		return luaL_error(L, "checkpointLoad must have a filename as the first argument");
	}
	
	const char* fn = lua_tostring(L, 1);
	FILE* f = fopen(fn, "r");
	
	if(!f)
	{
		return luaL_error(L, "failed to open `%s' for reading", fn);
	}
	lua_pop(L, lua_gettop(L));
	
	
	char header[128];
	int n;

	sure_fread(header, 1,         128, f); //should be CHECKPOINT\0\0\0...
	sure_fread(    &n, sizeof(int), 1, f); //read number of variables

	for(int i=0; i<n; i++)
	{
		int size;
		if(!sure_fread(&size, sizeof(int), 1, f))
		{
			fclose(f);
			return luaL_error(L, "failed in read\n");
		}

		char* buf = (char*)malloc(size+1);
		if(!sure_fread(buf, 1, size, f))
		{
			fclose(f);
			return luaL_error(L, "failed in read\n");
		}
		
		importLuaVariable(L, buf, size);

		free(buf);
	}
	
	fclose(f);
	return n;
}



static void uudecode_chunk(const char* _input4, char* output3)
{
	char input4[4];
	memcpy(input4, _input4, 4);
	for(int i=0; i<4; i++)
		input4[i] -= ' ';
	output3[0] = ((input4[0] & b0011_1111) << 2) | ((input4[1] >> 4) & b0000_0011);
	output3[1] = ((input4[1] << 4) & b1111_0000) | ((input4[2] >> 2) & b0000_1111);
	output3[2] = ((input4[2] << 6) & b1100_0000) | (input4[3] & b0011_1111);
}

static int l_uudecode(lua_State* L, const char* data)
{
	const int data_size = strlen(data);
	int data_pos = 0;
	
	const char CR = '\r';
	const char LF = '\n';
	
	int bin_size = 1024;
	int mem_step = 1024;
	int bin_pos = 0;
	char* bindata = (char*)malloc(bin_size);
	
	// find begin clause
	int begin_pos = -1;
	for(int i=0; (i<(data_size-5)) && (begin_pos==-1); i++)
	{
		if(strncmp(data+i, "begin", 5) == 0)
		{
			begin_pos = i;
		}
	}
	
	if(begin_pos == -1)
	{
		free(bindata);
		return luaL_error(L, "Failed to find `begin' clause in uuencoded string");
	}
	
	// we will now skip the file permissions and file name, looking for 
	int data_start = -1;
	for(int i=begin_pos; (i<data_size) && (data_start == -1); i++)
	{
		if( (data[i] == LF) | (data[i] == CR))
			data_start = i;
	}
	
	if(data_start == -1)
	{
		free(bindata);
		return luaL_error(L, "Failed to find end of `begin' clause in uuencoded string");		
	}
	
	data_pos = data_start;
	
	while((data[data_pos] == LF || data[data_pos] == CR) && data_pos < data_size)
		data_pos++;
	
	int line = 0;
	int bytes_to_read;
	
	while(1)
	{
		// here we are, we have some data
		bytes_to_read = (data[data_pos] - ' ') & b0011_1111;
		data_pos++;
		line++;
		
		if(bytes_to_read == 0)
			break;
		
		if(bytes_to_read < 0 || bytes_to_read > 45 || bytes_to_read + data_pos + 1 > data_size)
		{
			free(bindata);
			return luaL_error(L, "Out of range value found in byte count on data line %i", line);		
		}
		
		
		if(bin_size <= bin_pos + bytes_to_read + 3) //3 == margin
		{ 
			bindata = (char*)realloc(bindata, bin_size+mem_step + 3);
			bin_size += mem_step;
			mem_step *= 2;
		}
		
		for(int i=0; i<bytes_to_read; i+=3)
		{
			uudecode_chunk(data + data_pos, bindata + bin_pos + i);
			data_pos += 4;
		}
		
		bin_pos += bytes_to_read;
		
		
		// ok. Now to advance past control characters (we'll accept the case where they aren't present.
		while((data[data_pos] == LF || data[data_pos] == CR) && data_pos < data_size)
			data_pos++;
	}
	
	
	if(bin_pos < 128 + (int)sizeof(int))
	{
		free(bindata);
		return luaL_error(L, "Decoded data too small (%d). Not large enough to store header and data count.", bin_pos);
	}
	
	// got the binary representation, time to decode
	int n;
	{
		const char* d = bindata;
		int pos = 0;
		
		if(memcmp(d, "CHECKPOINT", 10) != 0)
		{
			free(bindata);
			return luaL_error(L, "Failed to find CHECKPOINT header in decoded data");
		}
		
		memcpy(&n, d+128, sizeof(int));
		
		pos += 128 + sizeof(int);
		
		for(int i=0; i<n; i++)
		{
			int size;
			memcpy(&size, d+pos, sizeof(int));
			pos += sizeof(int);
			
			importLuaVariable(L, (char*)d+pos, size); //yuck
			pos += size;
		}
	}

	free(bindata);
	return n;
}


static int l_checkpoint_load_from_string(lua_State* L)
{
	if(!lua_isstring(L, 1))
		return luaL_error(L, "checkpointFromString requires a string with uuencoded data.");
	
	const char* s = lua_tostring(L, 1);
	
	return l_uudecode(L, s);
}

#include "checkpointer.h"
#include "checkpointer_luafuncs.h"
void registerCheckPoint(lua_State* L)
{
	lua_pushcfunction(L, l_checkpoint_save);
	lua_setglobal(L, "checkpointSave");
	
	lua_pushcfunction(L, l_checkpoint_load);
	lua_setglobal(L, "checkpointLoad");

	lua_pushcfunction(L, l_checkpoint_save_to_string);
	lua_setglobal(L, "checkpointToString");

	lua_pushcfunction(L, l_checkpoint_load_from_string);
	lua_setglobal(L, "checkpointFromString");

}

#ifdef _CREATE_LIBRARY

#include "checkpoint_luafuncs.h"
#include "info.h"

#ifndef CHECKPOINT_API
#define CHECKPOINT_API
#endif

extern "C"
{
CHECKPOINT_API int lib_register(lua_State* L);
CHECKPOINT_API int lib_version(lua_State* L);
CHECKPOINT_API const char* lib_name(lua_State* L);
CHECKPOINT_API int lib_main(lua_State* L);
}

CHECKPOINT_API int lib_register(lua_State* L)
{
	registerCheckPoint(L);
	
	if(luaL_dostring(L, __checkpoint_luafuncs()))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
	}
	
	checkpointer_register(L);

	return 0;
}

CHECKPOINT_API int lib_version(lua_State* L)
{
	return __revi;
}

CHECKPOINT_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "CheckPoint";
#else
	return "CheckPoint-Debug";
#endif
}

CHECKPOINT_API int lib_main(lua_State* L)
{
	return 0;
}

#endif
