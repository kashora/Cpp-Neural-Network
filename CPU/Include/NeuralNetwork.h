#pragma once

#include <vector>
#include <iostream>
#include "HiddenLayer.h"
#include "Matrix.h"
using namespace std;
class NeuralNetwork {
private:

    int nInputs;
    int nOutputs;
    int nHiddenLayers;
    double lr;
    vector<HiddenLayer*> hiddenLayers;
    Matrix outputWeights;
    Matrix outputBiases;

    Matrix* hiddenLayersActivations;

    void (*outputActivFunct)(Matrix&);

    void initOutputLayer();
    Matrix calculateOutputError(Matrix& outputs, Matrix& expectedOutputs);
    void backPropagate(Matrix &inputs, Matrix &outputs, Matrix &expectedOutputs);
    void TrainBatch(Matrix& inputs, Matrix& expectedOutputs);
    void ValidateBatch(Matrix& inputs, Matrix& expectedOutputs, int&);


public:
    NeuralNetwork(int nInputs, int nOutputs, double lr);
    Matrix feedForward(Matrix& inputs);
    void addLayer(int nNeurons, std::string activationFunction);
	void Train(vector<vector<double>>& inputs, vector<int>& expectedOutputs, int epochs, int batchSize);
    void Validate(vector<vector<double>>& inputs, vector<int>& expectedOutputs, int batchSize);
};


