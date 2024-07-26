#include "Matrix.h"
#include <iostream>


const void Matrix::print() {
	// set the precision to 2 decimal places
	std::cout.precision(2);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			// use iomanip to seperate
			std::cout << std::fixed << data[i][j] << " ";

		}
		std::cout << std::endl;
	}
}

Matrix& Matrix::copy() {
	Matrix* newMatrix = new Matrix(rows, cols, false);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			newMatrix->data[i][j] = data[i][j];
		}
	}
	return *newMatrix;
}

Matrix Matrix::transpose() {
	Matrix newMatrix(cols, rows, false);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			newMatrix[j][i] = data[i][j];
		}
	}
	return newMatrix;
}


double getRandom() {
	// use a normal distribution with mean 0 and small standard deviation.
	return (rand() % 1000) / 1000.0;
}

Matrix::Matrix(int rows, int cols, bool is_random) {
	data = nullptr;
	if (rows == 0 || cols == 0) {

		throw std::invalid_argument("No enough data");
		return;
	}
	Generator gen(0, 0.1, -0.2, 0.2);
	this->rows = rows;
	this->cols = cols;
	this->data = new double* [rows];
	for (int i = 0; i < rows; i++) {
		this->data[i] = new double[cols];
		for (int j = 0; j < cols; j++) {
			if (is_random) {
				this->data[i][j] = gen();
			}
			else {
				this->data[i][j] = 0;
			}
		}
	}
}

Matrix::Matrix(const Matrix& other) {
	this->rows = other.rows;
	this->cols = other.cols;
	this->data = new double* [rows];
	for (int i = 0; i < rows; i++) {
		this->data[i] = new double[cols];
		for (int j = 0; j < cols; j++) {
			this->data[i][j] = other.data[i][j];
		}
	}
}

Matrix& Matrix::operator=(const Matrix& other) {
	if (this == &other) {
		return *this;
	}
	if (data != nullptr) {
		for (int i = 0; i < rows; i++) {
			delete[] data[i];
		}
		delete[] data;
	}
	this->rows = other.rows;
	this->cols = other.cols;
	this->data = new double* [rows];
	for (int i = 0; i < rows; i++) {
		this->data[i] = new double[cols];
		for (int j = 0; j < cols; j++) {
			this->data[i][j] = other.data[i][j];
		}
	}
	return *this;
}

Matrix::~Matrix() {
	if (data != nullptr) {
		for (int i = 0; i < rows; i++) {
			delete[] data[i];
		}
		delete[] data;
		data = nullptr; // Prevent double deletion
	}
}

Matrix Matrix::reduce() {
	Matrix B(rows, 1, false);
	for (int i = 0; i < rows; i++) {
		double sum = 0;
		for (int j = 0; j < cols; j++) {
			sum += data[i][j];
		}
		B[i][0] = sum;
	}
	return B;
}