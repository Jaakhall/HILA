//////////////////////////////////////////////////////////////////////////////
/// Test the standard gather here
//////////////////////////////////////////////////////////////////////////////

#include "defs.h"
#include "coordinates.h"
#include "lattice.h"
#include "field.h"


template <typename T>
// #pragma transformer dump ast
struct test_struct {
  T r[NDIM];

  using base_type = typename base_type_struct<T>::type;
};

using test_int = test_struct<int>;
using test_double = test_struct<double>;


template <typename T>
void gather_test() {

  extern lattice_struct * lattice;
  field<test_struct<T>> t;
  
  onsites(ALL) {
    coordinate_vector v = X.coordinates();
    foralldir(d) {
      t[X].r[d] = v[d];
    }
  }

  for (parity p : {EVEN,ODD,ALL}) {

    foralldir(d) {
      for (direction d2 : {d,-d}) {
      
        T diff = 0;
        int add;
        double sum1 = 0, sum2 = 0;  // use double to accumulate ints, should be accurate
        if (is_up_dir(d2)) add = 1; else add = -1;
        onsites(p) {
          T j = t[X+d2].r[d];
          T s = (t[X].r[d] + add + lattice->size(d)) % lattice->size(d);

          sum2 += t[X+d2].r[d] - lattice->size(d)/2;
          sum1 += t[X].r[d] - lattice->size(d)/2;

          T lv = s-j;
          T a = 0;
          foralldir(dir) if (dir != d) a+= t[X+d2].r[dir] - t[X].r[dir];
          
          if (lv != 0 || a != 0) {
            hila::output << "Error in gather test at " << X.coordinates() << " direction " << d2 
                         << " parity " << (int)p << '\n';
            hila::output << "Fetched element t[X+d2].r[d] = " << j << " should be " << s << " perp diff is " << a << '\n';
            hila::output << "This element - neighbour element:  ";
            for (int loop=0; loop<NDIM; loop++) hila::output << t[X].r[loop] << ' ';
            hila::output << " - ";
            for (int loop=0; loop<NDIM; loop++) hila::output << t[X+d2].r[loop] << ' ';
            
            hila::output << '\n';

            exit(-1);
          }
        }

        double s_result;
        if (p == ALL) 
          s_result = lattice->volume()/2;
        else 
          s_result = lattice->volume()/4;

        if (sum1 + s_result != 0.0) {
          output0 << "Error in sum reduction!  answer " << sum1 + s_result << " should be 0\n";
          exit(-1);
        }

        if (sum2 + s_result != 0.0) {
          output0 << "Error in neighbour sum reduction!  answer " << sum2 + s_result << " should be 0\n";
          exit(-1);
        }


        t.mark_changed(ALL);  // foorce fetching, test it too

#ifdef VECTORIZED
        // above is not vectorized, so do it also in vec way

        
        diff = 0;
        sum1 = sum2 = 0;
        onsites(p) {
          T j = t[X+d2].r[d];
          T s = (t[X].r[d] + add + lattice->size(d)) % lattice->size(d);

          diff += s-j;
          sum1 += t[X].r[d] - lattice->size(d)/2;
          sum2 += t[X+d2].r[d] - lattice->size(d)/2;
        }     
     
        if (diff != 0) {
          hila::output << "Vectorized std gather test error! Node " << mynode() 
                       << " Parity " << parity_name(p) << " direction " << (unsigned)d2 << '\n';
          exit(-1);
        }

        if (sum1 + s_result != 0.0) {
          output0 << "Error in vector sum reduction!  answer " << sum1 + s_result << " should be 0\n";
          exit(-1);
        }

        if (sum2 + s_result != 0.0) {
          output0 << "Error in vector neighbour sum reduction!  answer " << sum2 + s_result << " should be 0\n";
          exit(-1);
        }
        
        t.mark_changed(ALL);
#endif
      }
    }
  }
}



void test_std_gathers()
{
  gather_test<int>();
  // gather_test<int64_t>();
}

