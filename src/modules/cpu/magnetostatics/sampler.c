#include <stdio.h>
#include "gamma_ab_v.h"


void functionloop()
{
  int num = 10000, i;
  double d1,l1,w1,d2,l2,w2,x,y,z;
  double valuexx,valuexy,valuexz,valueyy,valueyz,valuezz,valueyx,valuezy,valuezx;
  for(i = 0; i < num; i++) //This was just to test the speed, the results are simply from the last time run
  {
  d1 = 4.0;
  l1 = 5.0;
  w1 = 6.0; 
  d2 = 1.0;
  l2 = 2.0;
  w2 = 3.0; 
  x = 10.0;
  y = 7.5;
  z = -10;
  valuexx = gamma_xx_v(x,y,z,d1,l1,w1,d2,l2,w2);
  valuexy = gamma_xy_v(x,y,z,d1,l1,w1,d2,l2,w2);
  valuexz = gamma_xz_v(x,y,z,d1,l1,w1,d2,l2,w2);
  valueyy = gamma_yy_v(x,y,z,d1,l1,w1,d2,l2,w2);
  valueyz = gamma_yz_v(x,y,z,d1,l1,w1,d2,l2,w2);
  valuezz = gamma_zz_v(x,y,z,d1,l1,w1,d2,l2,w2);
  valueyx = gamma_yx_v(x,y,z,d1,l1,w1,d2,l2,w2);
  valuezy = gamma_zy_v(x,y,z,d1,l1,w1,d2,l2,w2);
  valuezx = gamma_zx_v(x,y,z,d1,l1,w1,d2,l2,w2);
  }
  printf("x = %2.5g, y = %2.5g, z = %2.5g\n", x,y,z);
  printf("Computed integral xx = %0.10g\n", valuexx);
  printf("Computed integral xy = %0.10g\n", valuexy);
  printf("Computed integral xz = %0.10g\n", valuexz);
  printf("Computed integral yy = %0.10g\n", valueyy);
  printf("Computed integral yz = %0.10g\n", valueyz);
  printf("Computed integral zz = %0.10g\n", valuezz);
  printf("Computed integral yx = %0.10g\n", valueyx);
  printf("Computed integral zy = %0.10g\n", valuezy);
  printf("Computed integral zx = %0.10g\n", valuezx);


  printf("Computed sum = %0.10g\n", valuexx+valuexy+valuexz+valueyy+valueyz+valuezz+valueyx+valuezy+valuezx);
  //return 0;
}

int main(int argc, char *argv[])
{
  int num = 2, i;
  for(i = 0; i < num; i++)
  {
    functionloop();
  }
}


