#pragma once
#include <string>
#include "Matrix.h"


class HiddenLayer {
private:
    int nNeurons;
    int nInputs;
    Matrix weights;
    Matrix biases;
    std::string name;

public:
    double (*activationFunction)(double);
    double (*activationDerivative)(double);
    HiddenLayer(int numNeurons, int numInputs, std::string activationFunction, std::string name = "test");
    

    Matrix getWeights() { return weights; }
    Matrix getBiases() { return biases; }
    int getNumNeurons() { return nNeurons; }

    void updateWeights(Matrix weightChange, int NumOfSamples);
    void updateBiases(Matrix biasChange);

    HiddenLayer(const HiddenLayer& other);
    HiddenLayer& operator=(const HiddenLayer& other);
    ~HiddenLayer();
};

