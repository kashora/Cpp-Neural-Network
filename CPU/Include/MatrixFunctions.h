#pragma once
#include "Matrix.h"

Matrix MulMatrix(const Matrix& A, const Matrix& B);
Matrix MulMatrix(const Matrix& A, const double& scalar);
Matrix AddMatrix(const Matrix& A, const Matrix& B);
Matrix SubMatrix(const Matrix& A, const Matrix& B);
Matrix MulAccum(const Matrix& A, const Matrix& B, const Matrix& C);
Matrix HadamardProduct(const Matrix& A, const Matrix& B);
Matrix Transpose(const Matrix& A);

Matrix applyFunction(const Matrix& A, double (*function)(double));

