
#include "../plumbing/defs.h"
#include "../datatypes/cmplx.h"
#include "../plumbing/field.h"

// extern field<int> glob;

cmplx<double> d(cmplx<double> x) {return x;}
cmplx<double> e(cmplx<double> x) {return d(x);}
cmplx<double> f(cmplx<double> x) {return e(x);}

int main()
{
  
  field<cmplx<double>> a;
  field<double> t;
  t[ALL] = 1.0;

  onsites(ALL) {
    cmplx<double> d(2,2);
    a[X] = d;
  }

  double sum=0;
  onsites(ALL) {
    a[X] += a[X];
    sum += a[X].re;
  }
  
  return 0;
}
