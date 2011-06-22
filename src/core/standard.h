/*
------------------------------------------------------------------------------
Standard definitions and types, Bob Jenkins
------------------------------------------------------------------------------
*/
#ifndef STANDARD_H
# define STANDARD_H

#include <ctype.h>


// typedef  unsigned long long  ub8;
typedef  __u_quad_t  ub8;
#define UB8MAXVAL 0xffffffffffffffffLL
#define UB8BITS 64
typedef   __quad_t  sb8;
// typedef    signed long long  sb8;
#define SB8MAXVAL 0x7fffffffffffffffLL


// typedef  unsigned long  int  ub4;   /* unsigned 4-byte quantities */
typedef  __uint32_t ub4;
#define UB4MAXVAL 0xffffffff
typedef   __int32_t sb4;
// typedef    signed long  int  sb4;
#define UB4BITS 32
#define SB4MAXVAL 0x7fffffff
// typedef  unsigned short int  ub2;
typedef  __uint16_t  ub2;
#define UB2MAXVAL 0xffff
#define UB2BITS 16
// typedef    signed short int  sb2;
typedef   __int16_t  sb2;
#define SB2MAXVAL 0x7fff
// typedef  unsigned       char ub1;
typedef   __uint8_t  ub1;
#define UB1MAXVAL 0xff
#define UB1BITS 8
typedef    __int8_t  sb1;
// typedef    signed       char sb1;   /* signed 1-byte quantities */
#define SB1MAXVAL 0x7f
typedef                 int  word;  /* fastest type available */

#define bis(target,mask)  ((target) |=  (mask))
#define bic(target,mask)  ((target) &= ~(mask))
#define bit(target,mask)  ((target) &   (mask))
#ifndef min
# define min(a,b) (((a)<(b)) ? (a) : (b))
#endif /* min */
#ifndef max
# define max(a,b) (((a)<(b)) ? (b) : (a))
#endif /* max */
#ifndef align
# define align(a) (((ub4)a+(sizeof(void *)-1))&(~(sizeof(void *)-1)))
#endif /* align */
#define TRUE  1
#define FALSE 0
#define SUCCESS 0  /* 1 on VAX */

#endif /* STANDARD_H */
