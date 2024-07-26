#pragma once

#include <math.h>
#include <string>


double sigmoid(double x);
double sigmoidDerivative(double x);


double ReLU(double x);
double ReLUDerivative(double x);


double tanhAct(double x);
double tanhDerivative(double x);

void softmax(Matrix &A);



void getActivationFunction(std::string activationFunction, double (*&ActivFunct)(double), double (*&ActivFunctDirev)(double));
