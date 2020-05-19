#ifndef __DIRAC_WILSON_H__
#define __DIRAC_WILSON_H__

#include "../../plumbing/defs.h"
#include "../../datatypes/cmplx.h"
#include "../../datatypes/general_matrix.h"
#include "../../datatypes/wilson_vector.h"
#include "../../plumbing/field.h"



template<typename vector>
field<half_Wilson_vector<vector>> wilson_dirac_temp_vector[NDIM];



template<typename vector, typename matrix>
inline void dirac_wilson_hop(
  const field<matrix> *gauge, const double kappa,
  const field<Wilson_vector<vector>> &v_in,
  field<Wilson_vector<vector>> &v_out,
  parity par, int sign)
{
  field<half_Wilson_vector<vector>> (&vtemp)[NDIM] = wilson_dirac_temp_vector<vector>;
  foralldir(dir)
    vtemp[dir].copy_boundary_condition(v_in);

  // Run neighbour fetches and multiplications
  foralldir(dir){
    direction odir = opp_dir( (direction)dir );
    // First multiply the by conjugate before communicating
    onsites(opp_parity(par)){
      half_Wilson_vector<vector> h(v_in[X], dir, -sign);
      vtemp[dir][X] = gauge[dir][X].conjugate()*h;
    }

    vtemp[dir].set_boundary_condition(dir, v_in.get_boundary_condition(dir));
    vtemp[dir].start_get(odir);
  }
  foralldir(dir){
    direction odir = opp_dir( (direction)dir );
    onsites(par){
      half_Wilson_vector<vector> h1(v_in[X+dir], dir, sign);
      v_out[X] = v_out[X] 
               - (kappa*gauge[dir][X]*h1).expand(dir, sign)
               - (kappa*vtemp[dir][X-dir]).expand(dir, -sign);
    }
  }
}



template<typename vector>
inline void dirac_wilson_diag(
  const field<Wilson_vector<vector>> &v_in,
  field<Wilson_vector<vector>> &v_out,
  parity par)
{
  v_out[par] += v_in[X];
}


template<typename vector>
inline void dirac_wilson_diag_inverse(
  field<Wilson_vector<vector>> &v,
  parity par)
{}



template<typename vector, typename matrix>
inline void dirac_wilson_calc_force(
  const field<matrix> *gauge,
  const double kappa,
  const field<Wilson_vector<vector>> &chi,
  const field<Wilson_vector<vector>> &psi,
  field<matrix> (&out)[NDIM],
  parity par,
  int sign)
{
  field<half_Wilson_vector<vector>> (&vtemp)[NDIM] = wilson_dirac_temp_vector<vector>;
  vtemp[0].copy_boundary_condition(chi);
  vtemp[1].copy_boundary_condition(chi);
  
  foralldir(dir){
    onsites(opp_parity(par)){
      vtemp[0][X] = half_Wilson_vector<vector>(chi[X], dir, -sign);
    }
    onsites(par){
      vtemp[1][X] = half_Wilson_vector<vector>(psi[X], dir, sign);
    }

    out[dir][ALL] = 0;
    out[dir][par] = -kappa * (
      ( vtemp[0][X+dir].expand(dir,-sign) ).outer_product(psi[X])
    );
    out[dir][opp_parity(par)] = out[dir][X] - kappa * (
      ( vtemp[1][X+dir].expand(dir, sign) ).outer_product(chi[X])
    );

    out[dir][ALL] = gauge[dir][X]*out[dir][X];
  }
}



template<typename vector, typename matrix>
class dirac_wilson {
  private:
    double kappa;

    // Note array of fields, changes with the field
    field<matrix> (&gauge)[NDIM];
  
  public:

    using vector_type = Wilson_vector<vector>;
    using matrix_type = matrix;

    // Constructor: initialize mass and gauge
    dirac_wilson(dirac_wilson &d) : gauge(d.gauge), kappa(d.kappa) {}
    dirac_wilson(double k, field<matrix> (&U)[NDIM]) : gauge(U), kappa(k) {}

    // Applies the operator to in
    inline void apply( const field<vector_type> & in, field<vector_type> & out){
      out[ALL] = 0;
      dirac_wilson_diag(in, out, ALL);
      dirac_wilson_hop(gauge, kappa, in, out, ALL, 1);
    }

    // Applies the conjugate of the operator
    inline void dagger( const field<vector_type> & in, field<vector_type> & out){
      out[ALL] = 0;
      dirac_wilson_diag(in, out, ALL);
      dirac_wilson_hop(gauge, kappa, in, out, ALL, -1);
    }

    // Applies the derivative of the Dirac operator with respect
    // to the gauge field
    inline void force(const field<vector_type> & chi,  const field<vector_type> & psi, field<matrix> (&force)[NDIM], int sign=1){
      dirac_wilson_calc_force(gauge, kappa, chi, psi, force, ALL, sign);
    }
};


// Multiplying from the left applies the standard Dirac operator
template<typename vector, typename matrix>
field<Wilson_vector<vector>> operator* (dirac_wilson<vector, matrix> D, const field<Wilson_vector<vector>> & in) {
  field<Wilson_vector<vector>> out;
  D.apply(in, out);
  return out;
}

// Multiplying from the right applies the conjugate
template<typename vector, typename matrix>
field<Wilson_vector<vector>> operator* (const field<Wilson_vector<vector>> & in, dirac_wilson<vector, matrix> D) {
  field<Wilson_vector<vector>> out;
  D.dagger(in, out);
  return out;
}







template<typename vector, typename matrix>
class Dirac_Wilson_evenodd {
  private:
    double kappa;

    // Note array of fields, changes with the field
    field<matrix> (&gauge)[NDIM];
  public:

    using vector_type = Wilson_vector<vector>;
    using matrix_type = matrix;

    Dirac_Wilson_evenodd(Dirac_Wilson_evenodd &d) : gauge(d.gauge), kappa(d.kappa) {}
    Dirac_Wilson_evenodd(double k, field<matrix> (&U)[NDIM]) : gauge(U), kappa(k) {}


    // Applies the operator to in
    inline void apply( const field<vector_type> & in, field<vector_type> & out){
      out[ALL] = 0;
      dirac_wilson_diag(in, out, EVEN);

      dirac_wilson_hop(gauge, kappa, in, out, ODD, 1);
      dirac_wilson_diag_inverse(out, ODD);
      dirac_wilson_hop(gauge, kappa, out, out, EVEN, 1);
      out[ODD] = 0;
    }

    // Applies the conjugate of the operator
    inline void dagger( const field<vector_type> & in, field<vector_type> & out){
      out[ALL] = 0;
      dirac_wilson_diag(in, out, EVEN);

      dirac_wilson_hop(gauge, kappa, in, out, ODD, -1);
      dirac_wilson_diag_inverse(out, ODD);
      dirac_wilson_hop(gauge, kappa, out, out, EVEN, -1);
      out[ODD] = 0;
    }

    // Applies the derivative of the Dirac operator with respect
    // to the gauge field
    inline void force(const field<vector_type> & chi, const field<vector_type> & psi, field<matrix_type> (&force)[NDIM], int sign){
      field<matrix_type> force2[NDIM];
      field<vector_type> tmp;
      tmp.copy_boundary_condition(chi);

      tmp[ALL] = 0;
      dirac_wilson_hop(gauge, kappa, chi, tmp, ODD, -sign);
      dirac_wilson_diag_inverse(tmp, ODD);
      dirac_wilson_calc_force(gauge, kappa, tmp, psi, force, EVEN, sign);

      tmp[ALL] = 0;
      dirac_wilson_hop(gauge, kappa, psi, tmp, ODD, sign);
      dirac_wilson_diag_inverse(tmp, ODD);
      dirac_wilson_calc_force(gauge, kappa, chi, tmp, force2, ODD, sign);

      foralldir(dir)
        force[dir][ALL] = force[dir][X] + force2[dir][X];
    }
};


#endif