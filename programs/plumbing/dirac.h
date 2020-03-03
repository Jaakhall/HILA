#ifndef __DIRAC_H__
#define __DIRAC_H__

#include "../plumbing/defs.h"
#include "../datatypes/cmplx.h"
#include "../datatypes/general_matrix.h"
#include "../plumbing/field.h"


static int dirac_staggered_initialized = false;
static field<double> staggered_eta[NDIM];


inline void init_staggered_eta(){
  // Initialize the staggered eta field
  if(!dirac_staggered_initialized){
    foralldir(d){
      onsites(ALL){
        element<location> l = coordinates(X);
        element<int> sumcoord = 0;
        for(int d2=0;d2<d;d2++){
          sumcoord += l[d];
        }
        // +1 if sumcoord divisible by 2, -1 otherwise
        // If statements not yet implemented for vectors
        staggered_eta[d][X] = (sumcoord%2)*2-1; 
      }
    }
    dirac_staggered_initialized = true;
  }
}


template<typename mtype, typename vtype>
void dirac_stagggered(
  const mtype gauge[NDIM],
  const double mass,
  const vtype &v_in,
  vtype &v_out)
{
  static vtype vtemp[NDIM]; // A temporary vector field
  init_staggered_eta();
    
  // Start getting neighbours
  foralldir(dir){
    v_in.start_move(dir);
  }

  // Apply the mass diagonally
  v_out[ALL] = mass * v_in[X];

  // Run neighbour fetches and multiplications
  foralldir(dir){
    direction odir = opp_dir( (direction)dir );
    // First mulltiply the by conjugate before communicating the matrix
    vtemp[dir][ALL] = v_in[X]*gauge[dir][X].conjugate();
    vtemp[dir].start_move(odir);
  }
  foralldir(dir){
    direction odir = opp_dir( (direction)dir );
    v_out[ALL] += 0.5*staggered_eta[dir][X] * (
      v_in[X+dir]*gauge[dir][X] - vtemp[dir][X+odir]
    );
  }
}


template<typename mtype, typename vtype>
void dirac_stagggered_dagger(
  const mtype gauge[NDIM],
  const double mass,
  const vtype &v_in,
  vtype &v_out)
{
  static vtype vtemp[NDIRS]; // A temporary vector field
  init_staggered_eta();
    
  // Start getting neighbours
  foralldir(dir){
    direction odir = opp_dir( (direction)dir );
    v_in.start_move(dir);
  }

  // Apply the mass diagonally
  v_out[ALL] = mass * v_in[X];

  // Run neighbour fetches and multiplications
  foralldir(dir){
    direction odir = opp_dir( (direction)dir );
    // First mulltiply the by conjugate before communicating the matrix
    vtemp[dir][ALL] = v_in[X]*gauge[dir][X].conjugate();
    vtemp[dir].start_move(odir);
  }
  foralldir(dir){
    direction odir = opp_dir( (direction)dir );
    v_out[ALL] -= 0.5*staggered_eta[dir][X] * (
      v_in[X+dir]*gauge[dir][X] - vtemp[dir][X+odir]
    );
  }
}















#if (NDIM==4)
/// A staggered Dirac operator with one unrolled loop instead of
/// loops for all directions. Used in benchmarks.
template<typename mtype, typename vtype>
void dirac_stagggered_4dim(
  const mtype gauge[NDIM],
  const double mass,
  const vtype &v_in,
  vtype &v_out)
{
  static field<double> eta[NDIM]; // The staggered phase
  static vtype vtemp[NDIM]; // A temporary vector fields
  static bool initialized = false;

  // Initialize the staggered eta field
  if(!initialized){
    foralldir(d){
      onsites(ALL){
        element<location> l = coordinates(X);
        element<int> sumcoord = 0;
        for(int d2=0;d2<d;d2++){
          sumcoord += l[d];
        }
        // +1 if sumcoord divisible by 2, -1 otherwise
        // If statements not yet implemented for vectors
        eta[d][X] = (sumcoord%2)*2-1; 
      }
    }
    initialized = true;
  }

  onsites(ALL){
    vtemp[XUP][X] = v_in[X]*gauge[XUP][X].conjugate();
    vtemp[YUP][X] = v_in[X]*gauge[YUP][X].conjugate();
    vtemp[ZUP][X] = v_in[X]*gauge[ZUP][X].conjugate();
    vtemp[TUP][X] = v_in[X]*gauge[TUP][X].conjugate();
  }

  // Run neighbour fetches and multiplications
  onsites(ALL){
    v_out[X] = mass * v_in[X];
    v_out[X] += 0.5*eta[XUP][X]*v_in[X+XUP]*gauge[XUP][X];
    v_out[X] += 0.5*eta[YUP][X]*v_in[X+YUP]*gauge[YUP][X];
    v_out[X] += 0.5*eta[ZUP][X]*v_in[X+ZUP]*gauge[ZUP][X];
    v_out[X] += 0.5*eta[TUP][X]*v_in[X+TUP]*gauge[TUP][X];
    v_out[X] -= 0.5*eta[XUP][X]*vtemp[XUP][X+XDOWN];
    v_out[X] -= 0.5*eta[YUP][X]*vtemp[YUP][X+YDOWN];
    v_out[X] -= 0.5*eta[ZUP][X]*vtemp[ZUP][X+ZDOWN];
    v_out[X] -= 0.5*eta[TUP][X]*vtemp[TUP][X+TDOWN];
  }
}
#endif


#endif