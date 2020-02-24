#include "bench.h"

#ifndef MSIZE
#define MSIZE 3
#endif

#ifndef SEED
#define SEED 100
#endif

#define MADD(x) (MSIZE + x)

constexpr int mintime = CLOCKS_PER_SEC;

///////////////////////////////////////
// benchmark conjugate operations for 
// increasing matrix sizes starting 
// from msize.  
///////////////////////////////////////

int main(int argc, char **argv){
    int n_runs=1;
    double msecs;
    clock_t init, end;
    double timing;
    double sum;

    // Runs lattice->setup 
    bench_setup(argc, argv);
    // Seed rng generator
    seed_random(SEED);

    field<matrix<MADD(0),MADD(0), cmplx<double>> > matrix1;
    field<matrix<MADD(2),MADD(2), cmplx<double>> > matrix2;
    field<matrix<MADD(4),MADD(4), cmplx<double>> > matrix3;
    field<matrix<MADD(6),MADD(6), cmplx<double>> > matrix4;

    onsites(ALL){
      matrix<MADD(0),MADD(0), cmplx<double>> rand1;
      matrix<MADD(2),MADD(2), cmplx<double>> rand2;
      matrix<MADD(4),MADD(4), cmplx<double>> rand3;
      matrix<MADD(6),MADD(6), cmplx<double>> rand4;

      rand1.random();
      rand2.random();
      rand3.random();
      rand4.random();

      matrix1[X] = rand1;
      matrix2[X] = rand2;
      matrix3[X] = rand3;
      matrix4[X] = rand4;
    }

    // Time conj(matrix) * matrix * conj(matrix) 
    init = end = 0;
    for(n_runs=1; (end-init) < mintime; n_runs*=2){
      init = clock();
      for( int i=0; i<n_runs; i++){
          matrix1[ALL] = conj(matrix1[X])*matrix1[X]*conj(matrix1[X]);
      }
      synchronize();
      end = clock();
    }
    timing = (end - init) *1000.0 / ((double) CLOCKS_PER_SEC) / (double)n_runs;
    output0 << "matrix size " << (int) MSIZE << "*"  << (int) MSIZE << " : "<< timing << " ms \n";

    // Time conj(matrix) * matrix * conj(matrix) 
    init = end = 0;
    for(n_runs=1; (end-init) < mintime; n_runs*=2){
      init = clock();
      for( int i=0; i<n_runs; i++){
          matrix2[ALL] = conj(matrix2[X])*matrix2[X]*conj(matrix2[X]);
      }
      synchronize();
      end = clock();
    }
    timing = (end - init) *1000.0 / ((double) CLOCKS_PER_SEC) / (double)n_runs;
    output0 << "matrix size " << (int) MSIZE + 2 << "*"  << (int) MSIZE + 2 << " : "<< timing << " ms \n";

    // Time conj(matrix) * matrix * conj(matrix) 
    init = end = 0;
    for(n_runs=1; (end-init) < mintime; n_runs*=2){
      init = clock();
      for( int i=0; i<n_runs; i++){
          matrix3[ALL] = conj(matrix3[X])*matrix3[X]*conj(matrix3[X]);
      }
      synchronize();
      end = clock();
    }
    timing = (end - init) *1000.0 / ((double) CLOCKS_PER_SEC) / (double)n_runs;
    output0 << "matrix size " << (int) MSIZE + 4  << "*"  << (int) MSIZE + 4  << " : "<< timing << " ms \n";

    init = end = 0;
    for(n_runs=1; (end-init) < mintime; n_runs*=2){
      init = clock();
      for( int i=0; i<n_runs; i++){
          matrix4[ALL] = conj(matrix4[X])*matrix4[X]*conj(matrix4[X]);
      }
      synchronize();
      end = clock();
    }
    timing = (end - init) *1000.0 / ((double) CLOCKS_PER_SEC) / (double)n_runs;
    output0 << "matrix size " << (int) MSIZE + 6 << "*"  << (int) MSIZE + 6 << " : "<< timing << " ms \n";


    //------------------------------------------------

    // Time conj(matrix) * matrix * conj(matrix) 
    init = end = 0;
    for(n_runs=1; (end-init) < mintime; n_runs*=2){
      init = clock();
      for( int i=0; i<n_runs; i++){
          matrix1[ALL] = matrix1[X].conjugate()*matrix1[X]*matrix1[X].conjugate();
      }
      synchronize();
      end = clock();
    }
    timing = (end - init) *1000.0 / ((double) CLOCKS_PER_SEC) / (double)n_runs;
    output0 << "matrix size " << (int) MSIZE << "*"  << (int) MSIZE << " : "<< timing << " ms \n";

    // Time matrix) * .conjugate()atrix * matrix) 
    init = end = 0;
    for(n_runs=1; (end-init) < mintime; n_runs*=2){
      init = clock();
      for( int i=0; i<n_runs; i++){
          matrix2[ALL] = matrix2[X].conjugate()*matrix2[X]*matrix2[X].conjugate();
      }
      synchronize();
      end = clock();
    }
    timing = (end - init) *1000.0 / ((double) CLOCKS_PER_SEC) / (double)n_runs;
    output0 << "matrix size " << (int) MSIZE + 2 << "*"  << (int) MSIZE + 2 << " : "<< timing << " ms \n";

    // Time matrix) * .conjugate()atrix * matrix) 
    init = end = 0;
    for(n_runs=1; (end-init) < mintime; n_runs*=2){
      init = clock();
      for( int i=0; i<n_runs; i++){
          matrix3[ALL] = matrix3[X].conjugate()*matrix3[X]*matrix3[X].conjugate();
      }
      synchronize();
      end = clock();
    }
    timing = (end - init) *1000.0 / ((double) CLOCKS_PER_SEC) / (double)n_runs;
    output0 << "matrix size " << (int) MSIZE + 4  << "*"  << (int) MSIZE + 4  << " : "<< timing << " ms \n";

    init = end = 0;
    for(n_runs=1; (end-init) < mintime; n_runs*=2){
      init = clock();
      for( int i=0; i<n_runs; i++){
          matrix4[ALL] = matrix4[X].conjugate()*matrix4[X]*matrix4[X].conjugate();
      }
      synchronize();
      end = clock();
    }
    timing = (end - init) *1000.0 / ((double) CLOCKS_PER_SEC) / (double)n_runs;
    output0 << "matrix size " << (int) MSIZE + 6 << "*"  << (int) MSIZE + 6 << " : "<< timing << " ms \n";

    finishrun();
}


