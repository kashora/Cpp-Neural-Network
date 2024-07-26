#include "Matrix.h"
#include "Matrix.h"
#include "HiddenLayer.h"
#include "NeuralNetwork.h"
#include "ActivationFunctions.h"

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}
double sigmoidDerivative(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}


double ReLU(double x) {
    return x > 0 ? x : 0;
}
double ReLUDerivative(double x) {
    return x > 0 ? 1 : 0;
}


double tanhAct(double x) {
    return tanh(x);
}

double tanhDerivative(double x) {
    return 1 - pow(tanh(x), 2);
}

void softmax(Matrix &A) {
    for (int j = 0; j < A.getCols(); j++) {
        double sum = 0;
        double max = A[0][j];
        for (int i = 0; i < A.getRows(); i++) {
            sum += exp(A[i][j]);
            max = A[i][j] ? A[i][j] : max;
        }
        for (int i = 0; i < A.getRows(); i++) {
            A[i][j] = exp(A[i][j]) / sum;
        }
    }
}



void getActivationFunction(std::string activationFunction, double (*&ActivFunct)(double), double (*&ActivFunctDirev)(double)) {


    if (activationFunction == "sigmoid") {
        ActivFunct = sigmoid;
        ActivFunctDirev = sigmoidDerivative;
    }
    else if (activationFunction == "ReLU") {
        ActivFunct = ReLU;
        ActivFunctDirev = ReLUDerivative;
    }
    else if (activationFunction == "tanh") {
        ActivFunct = tanhAct;
        ActivFunctDirev = tanhDerivative;
    }
    else {
        if (activationFunction != "linear")
            printf("Invalid activation function. Using linear activation function instead.");
        ActivFunct = [](double x) {return x; };
        ActivFunctDirev = [](double x) {return static_cast<double>(1); };
    }

}
