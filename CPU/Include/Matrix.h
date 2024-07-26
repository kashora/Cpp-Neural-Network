#pragma once
#include <cstdlib>
#include <iostream>
#include <random>
class Matrix {
private:
    double** data;
    int rows;
    int cols;


public:

    Matrix(int rows=1, int cols=1, bool is_random=true);
    Matrix transpose();
    Matrix reduce();

    Matrix& copy();

    double*& operator[](int i) { return data[i]; }
    double** getData() { return data; }
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    double get(int i, int j) const {
        if (i >= rows || j >= cols || i < 0 || j < 0) {
            throw std::out_of_range("Index out of bounds");
        }
        return this->data[i][j];
    }
    const void print();

	Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    ~Matrix();
};

class Generator {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;
    double min;
    double max;
public:
    Generator(double mean, double stddev, double min, double max) :
        distribution(mean, stddev), min(min), max(max)
    {}

    double operator ()() {
        while (true) {
            double number = this->distribution(generator);
            if (number >= this->min && number <= this->max)
                return number;
        }
    }
};
