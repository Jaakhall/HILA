#ifndef GENERAL_MATRIX_H
#define GENERAL_MATRIX_H
#include<type_traits>
#include "operations.h"
#include "datatypes/cmplx.h"


// Do this macro here to ease "switching" between using
// mul_add operation and the normal sum
// THE mul_add METHOD SEEMS TO BE SLOWER?  
#define MUL_SUM(a, b, c) c += a*b
// #define MUL_SUM(a, b, c) c = mul_add(a, b, c)


//forward decl 
template <const int n, const int m, typename T>
class matrix;

//conjugate matrix class for special ops
template <const int n, const int m, typename T>
class conjugateMatrix {
  public: 
    conjugateMatrix(const matrix<n,m,T> & rhs) : ref(rhs) {};
    const matrix<n,m,T> & ref;
  private:
    conjugateMatrix(){}
    conjugateMatrix<n,m,T> & operator = (const conjugateMatrix & rhs){}
};

//transpose matrix class for special ops
template <const int n, const int m, typename T>
class transposeMatrix {
  public:
    transposeMatrix(const matrix<n,m,T> & rhs) : ref(rhs) {};
    const matrix<n,m,T> & ref;
  private:
    transposeMatrix(){}
    transposeMatrix<n,m,T> & operator = (const transposeMatrix & rhs){}
};



template <const int n, const int m, typename T>
class matrix {
  public:
  using base_type = typename base_type_struct<T>::type;

  T c[n][m];

  matrix() = default;

  matrix<n,m,T> operator-(void) { return -1*(*this); }

  template <typename scalart, std::enable_if_t<is_arithmetic<scalart>::value, int> = 0 >  
  #pragma hila loop_function
  matrix<n,m,T> & operator= (const scalart rhs) {
    static_assert(n==m, "rowdim != coldim : cannot assign diagonal from scalar!");
    for (int i=0; i<n; i++){
      c[i][i] = rhs;
      for (int j = 0; j < i; j++){
        c[i][j] = static_cast<T>(0.0); //ensure that matrix is zero initialized
        c[j][i] = static_cast<T>(0.0);
      }
    }
    return *this;
  }


  //copy constructor from scalar  
  template <typename scalart, std::enable_if_t<is_arithmetic<scalart>::value, int> = 0 >  
  #pragma hila loop_function
  matrix(const scalart rhs) {
    static_assert(n==m, "rowdim != coldim : cannot assign diagonal from scalar!");
    for (int i=0; i<n; i++) for (int j=0; j<m; j++) {
      if (i == j) c[i][j] = (rhs);
      else c[i][j] = (0);
    }
  }

  //*=, +=, -= operators
  #pragma hila loop_function
  matrix<n,m,T> & operator+=(const matrix<n,m,T> & rhs){
    for (int i = 0; i < n; i++) for (int j = 0; j < m; j++){
      c[i][j] += rhs.c[i][j]; 
    }
    return *this;
  }

  #pragma hila loop_function
  matrix<n,m,T> & operator-=(const matrix<n,m,T> & rhs){
    for (int i = 0; i < n; i++) for (int j = 0; j < m; j++){
      c[i][j] -= rhs.c[i][j]; 
    }
    return *this;
  }

  template <typename scalart, std::enable_if_t<is_arithmetic<scalart>::value, int> = 0 >
  #pragma hila loop_function
  matrix<n,m,T> & operator*=(const scalart rhs){
    T val;
    val=rhs;
    for (int i = 0; i < n; i++) for (int j = 0; j < m; j++){
      c[i][j]*=val;
    }
    return *this;
  }

  template<int p>
  #pragma hila loop_function
  matrix<n,m,T> & operator*=(const matrix<m,p,T> & rhs){
    static_assert(m==p, "can't assign result of *= to matrix A, because doing so would change it's dimensions");
    matrix<m,m,T> rhsTrans = rhs.transpose();
    matrix<n,m,T> res;
    for (int i = 0; i < n; i++) for (int j = 0; j < m; j++){
      res.c[i][j] = (0);
      for (int k = 0; k < m; k++){
        res.c[i][j] += (c[i][k] * rhsTrans.c[j][k]);
      }
    }
    for (int i = 0; i < n; i++) for (int j = 0; j < m; j++){
      c[i][j] = res.c[i][j];
    }
    return *this;
  }

  //numpy style matrix fill 
  template <typename scalart, std::enable_if_t<is_arithmetic<scalart>::value, int> = 0 > 
  #pragma hila loop_function
  matrix<n,m,T> & fill(const scalart rhs) {
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++){
      c[i][j] = (rhs);
    }
    return *this;
  }
  
  //return copy of transpose of this matrix
  #pragma hila loop_function
  matrix<m,n,T> transpose() const {
    matrix<m,n,T> res;
    for (int i=0; i<m; i++) for (int j=0; j<n; j++) {
      res.c[i][j] =  c[j][i];
    }
    return res;
  }

  //return copy of complex conjugate of this matrix
  #pragma hila loop_function
  matrix<m,n,T> conjugate() const {
    matrix<m,n,T> res;
    for (int i=0; i<m; i++) for (int j=0; j<n; j++) {
      res.c[i][j] =  conj(c[j][i]);
    }
    return res;
  }

  #pragma hila loop_function
  T trace() const {
    static_assert(n==m, "trace not defined for non square matrices!");
    T result = static_cast<T>(0);
    for (int i = 0; i < n; i++){
      result += c[i][i];
    }
    return result;
  }

  #pragma hila loop_function
  template <typename A=T, std::enable_if_t<is_arithmetic<A>::value, int> = 0 > 
  matrix<n, m, A> & random(){
    for (int i=0; i<n; i++) for (int j=0; j<m; j++) {
      c[i][j] = static_cast<T>(hila_random());
    }
    return *this;
  }

  #pragma hila loop_function
  template <typename A=T, std::enable_if_t<!is_arithmetic<A>::value, int> = 0 > 
  matrix<n, m, A> & random(){
    for (int i=0; i<n; i++) for (int j=0; j<m; j++) {
      c[i][j].random();
    }
    return *this;
  }

  auto norm_sq(){
    auto result = norm_squared(c[0][0]);
    for (int i=0; i<n; i++) for (int j=0; j<m; j++) if(i>0&&j>0) {
      result += norm_squared(c[i][j]);
    }
    return result;
  }

  inline T dot(const matrix<n, m, T> &rhs) const {
    T r = (0.0);
    for (int i=0; i<n; i++) for (int j=0; j<m; j++) {
      r += conj(c[i][j])*rhs.c[i][j];
    }
    return r;
  }

  std::string str() const {
    std::string text = "";
    for (int i=0; i<n; i++){
      for (int j=0; j<m; j++) {
        text + c[i][j].str() + " "; 
      }
      text + "\n"; 
    }
    return text;
  }
};

//templates needed for naive calculation of determinants

template<int n, int m, typename T> 
#pragma hila loop_function
matrix<n - 1, m - 1, T> Minor(const matrix<n, m, T> & bigger, int i, int j){
  matrix<n - 1, m - 1, T> result;
  int index = 0;
  for (int p = 0; p < n; p++) for (int l = 0; l < m; l++){
    if (p==i || l==j) continue;
    *(*(result.c) + index) = bigger.c[p][l];
    index++;
  }
  return result;
}

//determinant -> use LU factorization later 
template<int n, int m, typename T> 
#pragma hila loop_function
T det(const matrix<n, m, T> & mat){
  static_assert(n==m, "determinants defined only for square matrices");
  T result = 0.0;
  T parity = 1.0; //assumes that copy constructor from scalar has been defined for T 
  T opposite = -1.0; 
  for (int i = 0; i < n; i++){
    matrix<n - 1, m - 1, T> minor = Minor(mat, 0, i);
    result += parity*det(minor)*mat.c[0][i];
    parity*=opposite;
  }
  return result;
}

template<typename T> 
#pragma hila loop_function
T det(const matrix<2,2,T> & mat){
  return mat.c[0][0]*mat.c[1][1] - mat.c[1][0]*mat.c[0][1];
}

//matrix multiplication for 2 by 2 matrices ; 
template<typename T> 
#pragma hila loop_function
matrix<2,2,T> operator* (const matrix<2,2,T> &A, const matrix<2,2,T> &B) {
  matrix<2,2,T> res = 1;
  res.c[0][0] = A.c[0][0]*B.c[0][0] + A.c[0][1]*B.c[1][0];
  res.c[0][1] = A.c[0][0]*B.c[0][1] + A.c[0][1]*B.c[1][1];
  res.c[1][1] = A.c[1][0]*B.c[0][1] + A.c[1][1]*B.c[1][1];
  res.c[1][0] = A.c[1][0]*B.c[0][0] + A.c[1][1]*B.c[1][0];
  return res;
}

//matrix power 
template <int n, int m, typename T> 
#pragma hila loop_function
matrix<n,m,T> operator ^ (const matrix<n,m,T> & A, const int pow) {
  matrix<n,m,T> res;
  res = 1;
  for (int i = 0; i < pow; i++){
    res *= A;
  }
  return res;
}

//matrix * scalar 
template <int n, int m, typename T> 
#pragma hila loop_function
matrix<n,m,T> operator * (const matrix<n,m,T> & A, const T & B) {
  matrix<n,m,T> res;
  for (int i = 0; i < n; i++) for (int j = 0; j < m; j++){
    res.c[i][j] = A.c[i][j] * B;
  }
  return res;
}



//general matrix * matrix multiplication 
template <int n, int m, int p, typename T> 
#pragma hila loop_function
matrix<n,p,T> operator * (const matrix<n,m,T> &A, const matrix<m,p,T> &B) {
  matrix<n,p,T> res;
  for (int i = 0; i < n; i++) for (int j = 0; j < p; j++){
    res.c[i][j] = 0;
    for (int k = 0; k < m; k++){
      // res.c[i][j] += (A.c[i][k] * B.c[k][j]);
      MUL_SUM( A.c[i][k] , B.c[k][j] , res.c[i][j] );
    }
  }
  return res;
}

//multiplication for matrix * transpose matrix  
template <int n, int m, int p, typename T> 
#pragma hila loop_function
matrix<n,p,T> operator * (const matrix<n,m,T> & A, const transposeMatrix<p,m,T> & B) {
  matrix<n,p,T> res;
  for (int i = 0; i < n; i++) for (int j = 0; j < p; j++){
    res.c[i][j] = 0;
    for (int k = 0; k < m; k++){
      // res.c[i][j] += (A.c[i][k]*B.ref.c[j][k]);
      MUL_SUM(A.c[i][k] , B.ref.c[j][k], res.c[i][j]);
    }
  }
  return res;
}

//multiplication for transpose * matrix  
template <int n, int m, int p, typename T> 
#pragma hila loop_function
matrix<n,p,T> operator * (const transposeMatrix<m,n,T> & A, const matrix<m,p,T> & B) {
  matrix<n,p,T> res;
  for (int i = 0; i < n; i++) for (int j = 0; j < p; j++){
    res.c[i][j] = 0;
    for (int k = 0; k < m; k++){
      //res.c[i][j] += (A.ref.c[k][i]*B.c[k][j]);
      MUL_SUM(A.ref.c[k][i] , B.c[k][j], res.c[i][j]);
    }
  }
  return res;
}


//transpose * transpose
template <int n, int m, int p, typename T> 
#pragma hila loop_function
matrix<n,p,T> operator * (const transposeMatrix<m,n,T> & A, const transposeMatrix<p,m,T> & B) {
  matrix<n,p,T> res;
  for (int i = 0; i < n; i++) for (int j = 0; j < p; j++){
    res.c[i][j] = 0;
    for (int k = 0; k < m; k++){
      // res.c[i][j] += (A.ref.c[k][i]*B.ref.c[j][k]);
      MUL_SUM(A.ref.c[k][i] , B.ref.c[j][k], res.c[i][j]);
    }
  }
  return res;
}

//multiplication for matrix * conjugate matrix  
template <int n, int m, int p, typename T> 
#pragma hila loop_function
matrix<n,p,T> operator * (const matrix<n,m,T> & A, const conjugateMatrix<p,m,T> & B) {
  matrix<n,p,T> res;
  for (int i = 0; i < n; i++) for (int j = 0; j < p; j++){
    res.c[i][j] = 0;
    for (int k = 0; k < m; k++){
      // res.c[i][j] += (A.c[i][k]*conj(B.ref.c[j][k]));
      MUL_SUM( A.c[i][k] , conj(B.ref.c[j][k]), res.c[i][j] );
    }
  }
  return res;
}

//multiplication for conjugate * matrix  
template <int n, int m, int p, typename T> 
#pragma hila loop_function
matrix<n,p,T> operator * (const conjugateMatrix<m,n,T> & A, const matrix<m,p,T> & B) {
  matrix<n,p,T> res;
  for (int i = 0; i < n; i++) for (int j = 0; j < p; j++){
    res.c[i][j] = 0;
    for (int k = 0; k < m; k++){
      // res.c[i][j] += (conj(A.ref.c[k][i])*B.c[k][j]);
      MUL_SUM( conj(A.ref.c[k][i]) , B.c[k][j], res.c[i][j] );
    }
  }
  return res;
}


//conjugate * conjugate
template <int n, int m, int p, typename T> 
#pragma hila loop_function
matrix<n,m,T> operator * (const conjugateMatrix<m,n,T> & A, const conjugateMatrix<p,m,T> & B) {
  matrix<n,p,T> res;
  for (int i = 0; i < n; i++) for (int j = 0; j < p; j++){
    res.c[i][j] = 0;
    for (int k = 0; k < m; k++){
      res.c[i][j] += (conj(A.ref.c[k][i])*conj(B.ref.c[j][k]));
    }
  }
  return res;
}

//addition for matrix + conjugate matrix  
template <int n, int m, typename T> 
#pragma hila loop_function
matrix<n,m,T> operator + (const matrix<n,m,T> & A, const conjugateMatrix<m,n,T> & B) {
  matrix<n,m,T> res;
  for (int i = 0; i < n; i++) for (int j = 0; j < m; j++){
    res.c[i][j] = A.c[i][j] + conj(B.ref.c[j][i]);
  }
  return res;
}

//addition for conjugate + matrix  
template <int n, int m, typename T> 
#pragma hila loop_function
matrix<n,m,T> operator + (const conjugateMatrix<m,n,T> & A, const matrix<n,m,T> & B) {
  matrix<n,m,T> res;
  for (int i = 0; i < n; i++) for (int j = 0; j < m; j++){
    res.c[i][j] = conj(A.ref.c[j][i]) + B.c[i][j];
  }
  return res;
}

//addition for conjugate + conjugate
template <int n, int m, typename T> 
#pragma hila loop_function
matrix<n,m,T> operator + (const conjugateMatrix<m,n,T> & A, const conjugateMatrix<n,m,T> & B) {
  matrix<n,m,T> res;
  for (int i = 0; i < n; i++) for (int j = 0; j < m; j++){
      res.c[i][j] = conj(A.ref.c[j][i]) + conj(B.ref.c[j][i]);
  }
  return res;
}

//subtraction for matrix - conjugate matrix  
template <int n, int m, typename T> 
#pragma hila loop_function
matrix<n,m,T> operator - (const matrix<n,m,T> & A, const conjugateMatrix<m,n,T> & B) {
  matrix<n,m,T> res;
  for (int i = 0; i < n; i++) for (int j = 0; j < m; j++){
    res.c[i][j] = A.c[i][j] - conj(B.ref.c[j][i]);
  }
  return res;
}

//subtraction for conjugate - matrix  
template <int n, int m, typename T> 
#pragma hila loop_function
matrix<n,m,T> operator - (const conjugateMatrix<m,n,T> & A, const matrix<n,m,T> & B) {
  matrix<n,m,T> res;
  for (int i = 0; i < n; i++) for (int j = 0; j < m; j++){
    res.c[i][j] = conj(A.ref.c[j][i]) - B.c[i][j];
  }
  return res;
}

//subtraction for conjugate - conjugate
template <int n, int m, typename T> 
#pragma hila loop_function
matrix<n,m,T> operator - (const conjugateMatrix<m,n,T> & A, const conjugateMatrix<n,m,T> & B) {
  matrix<n,m,T> res;
  for (int i = 0; i < n; i++) for (int j = 0; j < m; j++){
      res.c[i][j] = conj(A.ref.c[j][i]) - conj(B.ref.c[j][i]);
  }
  return res;
}


//component wise addition
template <int n, int m, typename T> 
#pragma hila loop_function
matrix<n,m,T> operator+ (const matrix<n,m,T> &A, const matrix<n,m,T> &B) {
  matrix<n,m,T> res;
  for (int i=0; i<n; i++) for (int j=0; j<m; j++) {
    res.c[i][j] =  A.c[i][j] + B.c[i][j];
  }
  return res;
}

//component wise subtraction
template <int n, int m, typename T> 
#pragma hila loop_function
matrix<n,m,T> operator- (const matrix<n,m,T> &A, const matrix<n,m,T> &B) {
  matrix<n,m,T> res;
  for (int i=0; i<n; i++) for (int j=0; j<m; j++) {
    res.c[i][j] =  A.c[i][j] - B.c[i][j];
  }
  return res;
}

// multiplication by a scalar
template <int n, int m, typename T, typename scalart, std::enable_if_t<is_arithmetic<scalart>::value, int> = 0 > 
#pragma hila loop_function
matrix<n,m,T> operator* (const matrix<n,m,T> &A, const scalart s) {
  matrix<n,m,T> res;
  for (int i=0; i<n; i++) for (int j=0; j<m; j++) {
    res.c[i][j] = s * A.c[i][j];
  }
  return res;
}

template <int n, int m, typename T, typename scalart, std::enable_if_t<is_arithmetic<scalart>::value, int> = 0 > 
#pragma hila loop_function
matrix<n,m,T> operator/ (const matrix<n,m,T> &A, const scalart s) {
  matrix<n,m,T> res;
  for (int i=0; i<n; i++) for (int j=0; j<m; j++) {
    res.c[i][j] = s / A.c[i][j];
  }
  return res;
}

template <int n, int m, typename T, typename scalart, std::enable_if_t<is_arithmetic<scalart>::value, int> = 0 > 
#pragma hila loop_function
matrix<n,m,T> operator*(const scalart s, const matrix<n,m,T> &A) {
  return operator*(A,s);
}


template <int n, int m, typename T>
#pragma hila loop_function
std::ostream& operator<<(std::ostream &strm, const matrix<n,m,T> &A) {
  for (int i=0; i<n; i++){
    strm << "\n"; 
    for (int j=0; j<m; j++) {
      strm << " " << A.c[i][j] << " "; 
    }
    strm << "\n"; 
  }
  strm << "\n";
  return strm;
}

template<int n, int m, typename T> 
inline transposeMatrix<n,m,T> trans(matrix<n,m,T> & ref){
  transposeMatrix<n,m,T> result(ref);
  return result;
}

template<int n, int m, typename T> 
inline conjugateMatrix<n,m,T> conj(matrix<n,m,T> & ref){
  conjugateMatrix<n,m,T> result(ref);
  return result;
}

template<int n, int m, typename T>
inline auto norm_squared(matrix<n,m,T> & rhs){
  auto result = norm_squared(rhs.c[0][0]);
  for (int i=0; i<n; i++) for (int j=0; j<m; j++) if(i>0&&j>0) {
    result += norm_squared(rhs.c[i][j]);
  }
  return result;
}

#endif