#include "MatrixFunctions.h"

Matrix MulMatrix(const Matrix& A, const Matrix& B) {
	
	Matrix C(A.getRows(), B.getCols(), false);
	for (int i = 0; i < A.getRows(); i++) {
		for (int j = 0; j < B.getCols(); j++) {
			for (int k = 0; k < A.getCols(); k++) {
				C[i][j] += A.get(i,k) * B.get(k,j);
			}
		}
	}
	return C;
}

Matrix MulMatrix(const Matrix& A, double const & scalar) {
	Matrix C(A.getRows(), A.getCols(), false);
	for (int i = 0; i < A.getRows(); i++) {
		for (int j = 0; j < A.getCols(); j++) {
			C[i][j] = A.get(i,j) * scalar;
		}
	}
	return C;
}

Matrix AddMatrix(const Matrix& A, const Matrix& B) {
	Matrix C(A.getRows(), A.getCols(), false);
	for (int i = 0; i < A.getRows(); i++) {
		for (int j = 0; j < A.getCols(); j++) {
			C[i][j] = A.get(i,j) + B.get(i,j);
		}
	}
	return C;
}

Matrix SubMatrix(const Matrix& A, const Matrix& B) {

	if (A.getRows() != B.getRows() || A.getCols() != B.getCols()) {
		throw std::invalid_argument("Matrix dimensions must match");
	}

	Matrix C(A.getRows(), A.getCols(), false);
	for (int i = 0; i < A.getRows(); i++) {
		for (int j = 0; j < A.getCols(); j++) {
			C[i][j] = A.get(i,j) - B.get(i,j);
		}
	}
	return C;
}

Matrix MulAccum(const Matrix& A, const Matrix& B, const Matrix& C) {
	Matrix D(C.getRows(), B.getCols(), false);
	for (int i = 0; i < A.getRows(); i++) {
		for (int j = 0; j < B.getCols(); j++) {
			D[i][j] = C.get(i, 0);
			for (int k = 0; k < A.getCols(); k++) {
				D[i][j] += A.get(i,k) * B.get(k,j);
			}
		}
	}
	return D;
}


Matrix HadamardProduct(const Matrix& A, const Matrix& B) {
	Matrix C(A.getRows(), A.getCols(), false);
	for (int i = 0; i < A.getRows(); i++) {
		for (int j = 0; j < A.getCols(); j++) {
			C[i][j] = A.get(i,j) * B.get(i,j);
		}
	}
	return C;
}

Matrix applyFunction(const Matrix& A, double(*function)(double)) {
	Matrix B(A.getRows(), A.getCols());
	for (int i = 0; i < B.getRows(); i++) {
		for (int j = 0; j < B.getCols(); j++) {
			B[i][j] = function(A.get(i,j));
		}
	}
	return B;
}

Matrix Transpose(const Matrix& A) {
	Matrix B(A.getCols(), A.getRows(), false);
	for (int i = 0; i < A.getRows(); i++) {
		for (int j = 0; j < A.getCols(); j++) {
			B[j][i] = A.get(i, j);
		}
	}
	return B;
}