#include <luabaseobject.h>
#include "fingerprint_luafuncs.h"
#include <string.h>
#include <stdlib.h>

#include <MurmurHash3.h>

static char num2hex(unsigned int i)
{
    if(i < 10)
        return i+'0';
    return i-10+'A';
}

static unsigned int hex2num(const char c)
{
    if(c >= '0' && c <= '9')
        return c-'0';

    if(c >= 'a' && c <= 'f')
        return c+10-'a';

    if(c >= 'A' && c <= 'F')
        return c+10-'A';

    return 0;
}

static void hash2hex(const unsigned char* hash, int bytes, char* hex, int hex_len)
{
    /*
    printf("HASH: ");
    for(int i=0; i<bytes; i++)
        printf("%02X ", hash[i]);
    printf("\n");
    */

    int j;
    for(int i=0; i<bytes; i++)
    {
        unsigned int hi  = (0xF0 & hash[i]) >> 4;
        unsigned int low = 0x0F & hash[i];

        //printf("%i %X %x %x\n", i, hash[i], hi, low);

        if(j+1<hex_len) {hex[j] = num2hex(hi); hex[j+1] = 0;}
        j++;
        if(j+1<hex_len) {hex[j] = num2hex(low);  hex[j+1] = 0;}
        j++;
    }
}

static void hex2hash(const char* hex, int hex_len, unsigned char* data, unsigned int bytes)
{
    unsigned int high_pos = (hex_len % 2) ^ 0x1;
    unsigned int data_pos = 0;

    for(unsigned int i=0; i<bytes; i++)
    {
        data[i] = 0;
    }

    for(int i=0; i<hex_len; i++)
    {
        data[data_pos] |=  hex2num(hex[i]) << high_pos * 4;

        high_pos ^= (0x1); // flips 1 to 0 and back

        data_pos += high_pos; // equivalent to: if(high_pos) data_pos++

    }
}

static int l_hash128(lua_State* L)
{
    const int bytes = 16;
    const char* data = lua_tostring(L, 1);

    int seed = 1000;
//    if(lua_isnumber(L, 2))
//        seed = lua_tointeger(L, 2);

    unsigned char hash[bytes];

    MurmurHash3_x86_128(data, strlen(data), seed, (void*)hash);
    
    char s[2*bytes+1];
    hash2hex(hash, bytes, s, 2*bytes+1);
    lua_pushstring(L, s);
    return 1;
}



static int l_hash32(lua_State* L)
{
    const int bytes = 4;
    const char* data = lua_tostring(L, 1);

    int seed = 1000;
    if(lua_isnumber(L, 2))
        seed = lua_tointeger(L, 2);

    unsigned char hash[bytes];

    MurmurHash3_x86_32(data, strlen(data), seed, (void*)hash);
    
    char s[2*bytes+1];
    hash2hex(hash, bytes, s, 2*bytes+1);
    lua_pushstring(L, s);
    return 1;
}


// The following is based on 
// 
//  http://cvsweb.openbsd.org/cgi-bin/cvsweb/src/usr.bin/ssh/key.c?rev=1.90&content-type=text/x-cvsweb-markup
//
// It is under a very free license with the statement:
//

/* Copyright (c) 1995 Tatu Ylonen <ylo@cs.hut.fi>, Espoo, Finland
 *
 * As far as I am concerned, the code I have written for this software
 * can be used freely for any purpose.  Any derived versions of this
 * software must be clearly marked as such, and if the derived work is
 * incompatible with the protocol description in the RFC file, it must be
 * called by a name other than "ssh" or "Secure Shell".
 */


// The following is the code that I'm copying, it converts a binary string into 
// ascii art

/*
 * Draw an ASCII-Art representing the fingerprint so human brain can
 * profit from its built-in pattern recognition ability.
 * This technique is called "random art" and can be found in some
 * scientific publications like this original paper:
 *
 * "Hash Visualization: a New Technique to improve Real-World Security",
 * Perrig A. and Song D., 1999, International Workshop on Cryptographic
 * Techniques and E-Commerce (CrypTEC '99)
 * sparrow.ece.cmu.edu/~adrian/projects/validation/validation.pdf
 *
 * The subject came up in a talk by Dan Kaminsky, too.
 *
 * If you see the picture is different, the key is different.
 * If the picture looks the same, you still know nothing.
 *
 * The algorithm used here is a worm crawling over a discrete plane,
 * leaving a trace (augmenting the field) everywhere it goes.
 * Movement is taken from dgst_raw 2bit-wise.  Bumping into walls
 * makes the respective movement vector be ignored for this turn.
 * Graphs are not unambiguous, because circles in graphs can be
 * walked in either direction.
 */

/*
 * Field sizes for the random art.  Have to be odd, so the starting point
 * can be in the exact middle of the picture, and FLDBASE should be >=8 .
 * Else pictures would be too dense, and drawing the frame would
 * fail, too, because the key type would not fit in anymore.
 */

//#define FLDBASE 8
//#define FLDSIZE_Y (FLDBASE + 1)
//#define FLDSIZE_X (FLDBASE * 2 + 1)

#define MIN(a,b) (a<b?a:b)
#define MAX(a,b) (a>b?a:b)

typedef unsigned char u_char;
typedef unsigned int  u_int;

static char *key_fingerprint_randomart(u_char *dgst_raw, u_int dgst_raw_len, const char* title, 
                                       const int FLDSIZE_X, const int FLDSIZE_Y)
{
    /*
     * Chars to be used after each other every time the worm
     * intersects with itself.  Matter of taste.
     */
    const char* augmentation_string = " .o+=*BOX@%&#/^SE";
    char*retval, *p;

    u_char** field = (u_char**)malloc(sizeof(u_char*) * FLDSIZE_X);
    
    for(int i=0; i<FLDSIZE_X; i++)
    {
        field[i] = (u_char*)malloc(FLDSIZE_Y);
        memset(field[i], 0, FLDSIZE_Y * sizeof(char));
    }
    
    //u_char field[FLDSIZE_X][FLDSIZE_Y];

    u_int i, b;
    int x, y;
    size_t len = strlen(augmentation_string) - 1;

    retval = (char*) calloc(1, (FLDSIZE_X + 3) * (FLDSIZE_Y + 2));

    /* initialize field */
    x = FLDSIZE_X / 2;
    y = FLDSIZE_Y / 2;

    /* process raw key */
    for (i = 0; i < dgst_raw_len; i++) {
        int input;
        /* each byte conveys four 2-bit move commands */
        input = dgst_raw[i];
        for (b = 0; b < 4; b++) {
            /* evaluate 2 bit, rest is shifted later */
            x += (input & 0x1) ? 1 : -1;
            y += (input & 0x2) ? 1 : -1;

            /* assure we are still in bounds */
            x = MAX(x, 0);
            y = MAX(y, 0);
            x = MIN(x, FLDSIZE_X - 1);
            y = MIN(y, FLDSIZE_Y - 1);

            /* augment the field */
            if (field[x][y] < len - 2)
                field[x][y]++;
            input = input >> 2;
        }
    }

    /* mark starting point and end point*/
    field[FLDSIZE_X / 2][FLDSIZE_Y / 2] = len - 1;
    field[x][y] = len;


    /* fill in retval */
    // header line:
    int tlen = strlen(title);
    if(tlen+2 > FLDSIZE_X)
    {
        snprintf(retval, FLDSIZE_X+3, "+[%.*s...]+", FLDSIZE_X - 5, title);
    }
    else
    {
        char dashes[FLDSIZE_X*2];
        for(int i=0; i<FLDSIZE_X*2-1; i++) 
            dashes[i] = '-';
        dashes[FLDSIZE_X*2] = 0;

        if(tlen == 0)
        {
            snprintf(retval, FLDSIZE_X+3, "+%.*s+", FLDSIZE_X, dashes);
        }
        else
        {
            int pre = 0;
            int post = 0;
            while(pre + post + tlen <= FLDSIZE_X-3)
            {
                pre++;
                if(pre + post + tlen <= FLDSIZE_X-3)
                    post++;
            }
            
            snprintf(retval, FLDSIZE_X+3, "+%.*s[%s]%.*s+", pre, dashes, title, post, dashes);
        }
    }

    p = strchr(retval, '\0');
    *p++ = '\n';

    /* output content */
    for (y = 0; y < FLDSIZE_Y; y++) {
        *p++ = '|';
        for (x = 0; x < FLDSIZE_X; x++)
            *p++ = augmentation_string[MIN(field[x][y], len)];
        *p++ = '|';
        *p++ = '\n';
    }

    /* output lower border */
    *p++ = '+';
    for (i = 0; i < FLDSIZE_X; i++)
        *p++ = '-';
    *p++ = '+';

    for(int i=0; i<FLDSIZE_X; i++)
        free(field[i]);
    free(field);


    return retval;
}

static int l_key_fp(lua_State* L)
{
    const char* txt_hex = lua_tostring(L, 1);

    int len = strlen(txt_hex);

    unsigned char* data = (unsigned char*)malloc(len*2);

    int nx = 17;
    int ny = 9;

    if(lua_isnumber(L, -2))
        nx = lua_tointeger(L, -2);
    if(lua_isnumber(L, -1))
        ny = lua_tointeger(L, -1);

    if(nx < 5)
        nx = 5;
    if(ny < 3)
        ny = 3;

    // make them odd
    nx += ((nx + 1) % 2);
    ny += ((ny + 1) % 2);

    hex2hash(txt_hex, len, data, len*2);

    char* fp;

    if(!lua_isnumber(L, 2) && lua_isstring(L, 2))
        fp = key_fingerprint_randomart(data, len*2, lua_tostring(L, 2), nx, ny);
    else
        fp = key_fingerprint_randomart(data, len*2, "", nx,ny);

    lua_pushstring(L, fp);
    free(fp);
    free(data);
    return 1;
}



#include "info.h"
extern "C"
{
int lib_register(lua_State* L);
int lib_version(lua_State* L);
const char* lib_name(lua_State* L);
int lib_main(lua_State* L);
}

int lib_register(lua_State* L)
{
    lua_pushcfunction(L, l_hash128);
    lua_setglobal(L, "fp_hash128");

    lua_pushcfunction(L, l_hash32);
    lua_setglobal(L, "fp_hash32");

    lua_pushcfunction(L, l_key_fp);
    lua_setglobal(L, "fp_hashFingerprint");

    const char* s = __fingerprint_luafuncs();

    if(luaL_dostringn(L, s, "fingerprint_luafuncs.lua"))
    {
        fprintf(stderr, "%s\n", lua_tostring(L, -1));
        return luaL_error(L, lua_tostring(L, -1));
    }

    return 0;
}

int lib_version(lua_State* L)
{
	return __revi;
}

const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Fingerprint";
#else
	return "Fingerprint-Debug";
#endif
}

#include "fingerprint_main.h"
int lib_main(lua_State* L)
{
    if(luaL_dostringn(L, __fingerprint_main(), "fingerprint_main.lua"))
    {
        fprintf(stderr, "%s\n", lua_tostring(L, -1));
        return luaL_error(L, lua_tostring(L, -1));
    }

    return 0;
}
