#include "test.h"

/////////////////////
/// test_case 2
/// Coverage:
/// - directions, onsites and foralldir environments
/// - operations between fields 
/// - foralldir env inside onsites
/// - referring to an array of fields in a loop
/// - calling a function with const parameters
///   - requiring communication of a const field
/// - calling a function from a loop
/////////////////////

template<typename A, typename B, typename C>
void sum_test_function(A &a, const B &b, const C &c){
    onsites(ALL){
        a[X] = b[X] + c[X+XUP];
    }
}

template<typename T>
T test_template_function(T a){
  return 2*a;
}

element<cmplx<double>> test_nontemplate_function(element<cmplx<double>> a){
  element<cmplx<double>> b = a;
  return 2*a;
}


int main(int argc, char **argv){

    //check that you can increment a direction correctly
    direction d = XUP;
    direction d2 = (direction) (NDIRS - 2);
    #if NDIM > 1
    d=next_direction(d); 
    d2=next_direction(d2);
    assert(d==YUP);
    assert(XUP==0);
    assert(d2==XDOWN);
    #endif

    double sum = 0;
    field<cmplx<double>> s1, s2, s3;
    field<cmplx<double>> s4[3];

    test_setup(argc, argv);

    // Test field assingment
    s1 = 0.0;
    s2 = 1.0;
    s3 = 1.0;

    // Test sum and move constructor
    s1 = s2 + s3;

    onsites(ALL){
        sum+=s1[X].re;
    }
    assert(sum==2*(double)lattice->volume() && "onsites reduction");
    s1=0; s2=0; s3=0;
    sum = 0;

    // Test field-parity expressions
    s1[ALL] = 0.0;
    s2[EVEN] = 1.0;
    s3[ODD] = 1.0;

    s1[ALL] = s2[X] + s3[X];

    onsites(ALL){
        sum+=s1[X].re;
    }
    assert(sum==(double)lattice->volume() && "test setting field with parity");

    // Test communication functions
    s1[ALL] = 0;

    assert(s1.is_allocated());
    assert(s1.is_initialized(EVEN));
    assert(s1.is_initialized(ODD));

    s1.mark_changed(ALL);

    // Check initial state of communication functions
    foralldir(d){
      assert(!s1.is_fetched(d, EVEN));
      assert(!s1.is_fetched(d, ODD));
      assert(!s1.is_fetched(d, ALL));
      assert(!s1.is_move_started(d, EVEN));
      assert(!s1.is_move_started(d, ODD));
      assert(!s1.is_move_started(d, ALL));
      assert(s1.move_not_done(d, EVEN) && "move not done initially");
      assert(s1.move_not_done(d, ODD) && "move not done initially");
      assert(s1.move_not_done(d, ALL) && "move not done initially");
    }

    // Test marking move started and fetched
    foralldir(d){
      s1.mark_move_started(d, EVEN);
      assert(s1.is_move_started(d, EVEN));
      assert(!s1.is_fetched(d, EVEN));
      assert(!s1.move_not_done(d, EVEN) && "move not done after starting");
      assert(!s1.is_fetched(d, ODD));
      assert(!s1.is_fetched(d, ALL));
      assert(!s1.is_move_started(d, ODD));
      assert(!s1.is_move_started(d, ALL));
      assert(s1.move_not_done(d, ODD));
      assert(s1.move_not_done(d, ALL));

      s1.mark_fetched(d, EVEN);
      assert(s1.is_fetched(d, EVEN));
      assert(!s1.is_move_started(d, EVEN));
      assert(!s1.move_not_done(d, EVEN));

      s1.mark_changed(ALL);

      s1.mark_move_started(d, ODD);
      assert(s1.is_move_started(d, ODD));
      assert(!s1.is_fetched(d, ODD));
      assert(!s1.move_not_done(d, ODD) && "move not done after starting");
      
      s1.mark_fetched(d, ODD);
      assert(s1.is_fetched(d, ODD));
      assert(!s1.is_move_started(d, ODD));
      assert(!s1.move_not_done(d, ODD));

      s1.mark_changed(ALL);

      s1.mark_move_started(d, ALL);
      assert(s1.is_move_started(d, ALL));
      assert(!s1.is_fetched(d, ALL));
      assert(!s1.move_not_done(d, ALL) && "move not done after starting");
      
      s1.mark_fetched(d, ALL);
      assert(s1.is_fetched(d, ALL));
      assert(!s1.is_move_started(d, ALL));
      assert(!s1.move_not_done(d, ALL));

      s1.mark_changed(ALL);
    }


    // Try setting an element on node 0
    coordinate_vector coord;
    foralldir(d) {
      coord[d] = 0;
    }
    s1.set_element(cmplx<double>(1), coord);
    cmplx<double> elem = s1.get_element(coord);
    assert(elem.re == 1 && elem.im==0);

    // Now try setting on a different node, if the lattice is split
    foralldir(d) {
      coord[d] = nd[d]-1;
    }
    s1.set_element(cmplx<double>(1), coord);
    elem = s1.get_element(coord);
    assert(elem.re == 1 && elem.im==0);


    // Now try actually moving the data
    foralldir(d) {
      coord[d] = 0;
    }
    foralldir(d){
      // Move data up in direction d
      s2[ALL] = s1[X-d];

      // Should still be on this node
      coordinate_vector coord2 = coord;
      coord2[d] += 1;
      cmplx<double> moved = s2.get_element(coord2);
      assert(elem.re == 1 && elem.im==0);

      // Move data down
      s2[ALL] = s1[X+d];

      // Now it may be on a different node
      coord2 = coord;
      coord2[d] = (coord[d] - 1 + nd[d]) % nd[d];
      moved = s2.get_element(coord2);
      if( elem.re != 1 || elem.im != 0 ){
        output0 << "Problem in communicating to direction " << d << "\n";
        output0 << "Received " << moved << "\n";
        assert(elem.re == 1 && elem.im==0);
      }
    }


    // Communication and copy test with full field
    foralldir(d){
      s1 = 1.0; s2 = 1.0; s3 = 1.0;
      double sum = 0;
      double sum2 = 0;
      onsites(EVEN){
        double a = s1[X+d].re;
        double b = s2[X].re;
        sum += a-b;
      }

      output0 << d << " " << sum << "\n";
	    assert(sum==0 && "Test communicating a filled field");


      s1 = 1.0; s2 = 1.0; s3 = 1.0; sum = 0; sum2 = 0;
      onsites(EVEN){
        sum += s2[X].re-s1[X+d].re;
        s2[X] -= 1.0;
        sum2 += s2[X].re;
      }

      output0 << d << " " << sum << " " << sum2 << "\n";

	    assert(sum==0 && "Reproduce write problem");
    }



    // Test starting communication manually

    s1[EVEN] = 1.0;
    s2[EVEN] = 1.0;
    s2[ODD] = -s1[X+XUP];
    s2.start_get(XUP,ODD);

    sum = 0;
    onsites(ALL){
	    sum += s2[X].re;
    }
	  assert(sum==0);

    // Test referring to an array of fields

    s4[0] = s1;
    s4[1] = s1;

    s4[2][ALL] = s4[0][X] - s4[1][X];

    sum = 0;
    onsites(ALL){
        sum += (s4[2][X]*s4[2][X]).re;
    }
    assert(sum == 0);

    //Test function call outside loop
    s1[ALL] = 0.0;
    s2[ALL] = 1.0;
    sum_test_function( s3, s1, s2 ); //s3 = s1 + s2
    onsites(ALL){
        element<double> diff = s3[X].re - 1.0;
        sum += diff*diff;
    }
    assert(sum == 0);

    //Test function calls in loop
    s1[ALL] = 1.0;
    s2[ALL] = 1.0;
    onsites(ALL){
      s1[X] = test_template_function(s1[X]);
      s2[X] = test_nontemplate_function(s2[X]);
    }
    onsites(ALL){
        element<double> diff1 = s1[X].re - 2.0;
        element<double> diff2 = s2[X].re - 2.0;
        sum += diff1*diff1 + diff2*diff2;
    }
    assert(sum == 0);


    // Test array reduction
    field<double> dfield;
    dfield[ALL] = 1;

#if NDIM == 4
    std::vector<double> arraysum(nd[TUP]);
    std::fill(arraysum.begin(), arraysum.end(), 0);

    onsites(ALL){
      element<coordinate_vector> l = X.coordinates();
      element<int> t = l[TUP];
      
      arraysum[t] += dfield[X];
    }
    
    for(int t=0; t<nd[TUP]; t++){
      assert(arraysum[t] == nd[XUP]*nd[YUP]*nd[ZUP]);
    }
#endif

    finishrun();
}
