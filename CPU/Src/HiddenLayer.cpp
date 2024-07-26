#include "Matrix.h"
#include "HiddenLayer.h"
#include "NeuralNetwork.h"
#include "ActivationFunctions.h"
#include "MatrixFunctions.h"

HiddenLayer::HiddenLayer(int nNeurons, int nInputs, std::string activationFunction, std::string name) : weights(nNeurons, nInputs, true), biases(nNeurons, 1, true) {
    this->nNeurons = nNeurons;
    this->nInputs = nInputs;
	this->name = name;

    getActivationFunction(activationFunction, this->activationFunction, this->activationDerivative);

}



void HiddenLayer::updateWeights(Matrix weightChange, int NumOfSamples) { 
	weightChange = MulMatrix(weightChange, 1.0 / NumOfSamples);
	weights = SubMatrix(weights, weightChange);
}
void HiddenLayer::updateBiases(Matrix biasChange) { 
	biases = SubMatrix(biases, biasChange.reduce());
}


HiddenLayer::HiddenLayer(const HiddenLayer& other) {
	nNeurons = other.nNeurons;
	nInputs = other.nInputs;
	weights = other.weights;
	biases = other.biases;
	name = other.name;
	activationFunction = other.activationFunction;
	activationDerivative = other.activationDerivative;
}

HiddenLayer& HiddenLayer::operator=(const HiddenLayer& other) {
	if (this == &other) {
		return *this;
	}
	nNeurons = other.nNeurons;
	nInputs = other.nInputs;
	weights = other.weights;
	biases = other.biases;
	name = other.name;
	activationFunction = other.activationFunction;
	activationDerivative = other.activationDerivative;
	return *this;
}

HiddenLayer::~HiddenLayer() {
	weights.~Matrix();
	biases.~Matrix();
}
