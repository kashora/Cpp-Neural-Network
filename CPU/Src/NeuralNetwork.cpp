#include "Matrix.h"
#include "HiddenLayer.h"
#include "NeuralNetwork.h"
#include "ActivationFunctions.h"
#include "MatrixFunctions.h"
#include <iostream>
using namespace std;


int x = 0;
NeuralNetwork::NeuralNetwork(int nInputs, int nOutputs, double lr) : outputBiases(nOutputs, 1), outputWeights(nOutputs, nInputs){
    this->nInputs = nInputs;
    this->nOutputs = nOutputs;
    this->nHiddenLayers = 0;
    this->outputActivFunct = softmax;
    this->lr = lr;
	hiddenLayersActivations = nullptr;
}

void NeuralNetwork::Train(vector<vector<double>>& inputs, vector<int>& expectedOutputs, int epochs, int batchSize) {
    hiddenLayersActivations = new Matrix[nHiddenLayers];
	for (int i = 0; i < epochs; i++) {
        cout << "Epoch: " << i + 1 << "/" << epochs << endl;
		for (int j = 0; j < inputs.size(); j += batchSize) {
            
			Matrix inputBatch(nInputs, batchSize, false);
			Matrix expectedOutputBatch(nOutputs, batchSize, false);
			for (int k = 0; k < batchSize && j+k<inputs.size(); k++) {
                x++;
				for (int l = 0; l < nInputs; l++) {
					inputBatch[l][k] = inputs[j + k][l];
				}
				expectedOutputBatch[expectedOutputs[j + k]][k] = 1;
			}
			TrainBatch(inputBatch, expectedOutputBatch);
            inputBatch.~Matrix();
            expectedOutputBatch.~Matrix();

            //cout << "Done: " << j + batchSize << "/" << inputs.size()<< endl;
		}
	}
}

void NeuralNetwork::Validate(vector<vector<double>>& inputs, vector<int>& expectedOutputs, int batchSize) {
    int CorrectPrediction = 0;
    for (int j = 0; j < inputs.size(); j += batchSize) {
        Matrix inputBatch(nInputs, batchSize, false);
        Matrix expectedOutputBatch(nOutputs, batchSize, false);
        for (int k = 0; k < batchSize && j + k < inputs.size(); k++) {
            x++;
            for (int l = 0; l < nInputs; l++) {
                inputBatch[l][k] = inputs[j + k][l];
            }
            expectedOutputBatch[expectedOutputs[j + k]][k] = 1;
        }
        
        ValidateBatch(inputBatch, expectedOutputBatch, CorrectPrediction);
        inputBatch.~Matrix();
        expectedOutputBatch.~Matrix();
    }
    cout << "Validation Completed\n";
    cout << "Number of correct predictions: " << CorrectPrediction << "/" << inputs.size() << endl;
    cout << "Total Accuracy: " << (1.0f * CorrectPrediction) / inputs.size();

}



vector<int> getHighestInCol(const Matrix& A) {
    vector<int> highest;
    for (int j = 0; j < A.getCols(); j++) {
        double max = A.get(0, j);
        int index = 0;
        for (int i = 1; i < A.getRows(); i++) {
            if (A.get(i, j) > max) {
                max = A.get(i, j);
                index = i;
            }
        }
        highest.push_back(index);
    }
    return highest;
}

void NeuralNetwork::ValidateBatch(Matrix& inputs, Matrix& expectedOutputs, int& CorrectPredictions) {
    Matrix predictedOutputs = feedForward(inputs);
    vector<int> highestPredicted = getHighestInCol(expectedOutputs);
    vector<int> highestExpected = getHighestInCol(predictedOutputs);

    for (int i = 0; i < highestExpected.size(); i++) {
        if (highestExpected.at(i) == highestPredicted.at(i))
            CorrectPredictions += 1;
    }
}




void NeuralNetwork::TrainBatch(Matrix& inputs, Matrix& expectedOutputs) {
    Matrix outputs = feedForward(inputs);
    backPropagate(inputs, outputs, expectedOutputs);

    outputs.~Matrix();
}




void NeuralNetwork::initOutputLayer() { // Will be called after all hidden layers are added (before training)
    outputWeights.~Matrix();
    Matrix OW(nOutputs, hiddenLayers[nHiddenLayers - 1]->getNumNeurons(), true);
    outputWeights = OW.copy();
    OW.~Matrix();
}



Matrix NeuralNetwork::feedForward(Matrix& inputs) {
    Matrix currentActivation = inputs;
    for (int i = 0; i < nHiddenLayers; i++) {

		currentActivation = MulAccum(hiddenLayers[i]->getWeights(), currentActivation, hiddenLayers[i]->getBiases());
		currentActivation = applyFunction(currentActivation, hiddenLayers[i]->activationFunction);
        hiddenLayersActivations[i] = currentActivation;
    }


    currentActivation = MulAccum(outputWeights, currentActivation, outputBiases);
    //
    //currentActivation.print();
    outputActivFunct(currentActivation);
	
	//currentActivation = applyFunction(currentActivation, sigmoid);
    //currentActivation.print();

    return currentActivation;
}

void NeuralNetwork::addLayer(int nNeurons, std::string activationFunction) {
    if (nHiddenLayers == 0) {
		HiddenLayer* temp = new HiddenLayer(nNeurons, nInputs, activationFunction, "Layer 0");
        hiddenLayers.push_back(temp);
    }
    else {
		HiddenLayer* temp = new HiddenLayer(nNeurons, hiddenLayers[nHiddenLayers - 1]->getNumNeurons(), activationFunction, "Layer " + nHiddenLayers);
        hiddenLayers.push_back(temp);
    }
    nHiddenLayers++;
    initOutputLayer();
}

/*
Matrix NeuralNetwork::calculateOutputError(Matrix& outputs, Matrix& expectedOutputs) {
    Matrix error = SubMatrix(outputs, expectedOutputs);
    return error;
}
*/
Matrix NeuralNetwork::calculateOutputError(Matrix& outputs, Matrix& expectedOutputs) {
    int batchSize = outputs.getCols();
    int numClasses = outputs.getRows();
    Matrix error(numClasses, batchSize, false);

    for (int i = 0; i < batchSize; i++) {
        double batchLoss = 0.0;
        for (int j = 0; j < numClasses; j++) {
            double y = expectedOutputs.get(j, i);
            double y_hat = outputs.get(j, i);

            // Clip predicted probabilities to avoid log(0)
            y_hat = std::max(1e-15, std::min(1.0 - 1e-15, y_hat));

            // Calculate the gradient of cross-entropy loss
            error[j][i] = y_hat - y;

            // Accumulate the loss (optional, for reporting purposes)
            if (y == 1.0) {
                batchLoss -= std::log(y_hat);
            }
        }
        // You can use batchLoss here if you want to report the average loss
    }

    // Normalize the gradient by batch size
    return MulMatrix(error, 1.0 / batchSize);
}

void printHighestInCol(const Matrix& A) {
	for (int j = 0; j < A.getCols(); j++) {
		double max = A.get(0,j);
		int index = 0;
		for (int i = 1; i < A.getRows(); i++) {
			if (A.get(i,j) > max) {
				max = A.get(i, j);
				index = i;
			}
		}
		std::cout << index << " ";
	}
    std::cout << endl;

}

void NeuralNetwork::backPropagate(Matrix &inputs, Matrix &outputs, Matrix &expectedOutputs) {
    Matrix error = calculateOutputError(outputs, expectedOutputs);
    
	if (3000<x) {
		//std::cout << "Input: " << std::endl;
		//inputs.print();
		std::cout << "Expected: " << std::endl;
        printHighestInCol(expectedOutputs);
		//expectedOutputs.print();
		std::cout << "Output: " << std::endl;
		printHighestInCol(outputs);
		//outputs.print();
		std::cout << "Error: " << std::endl;
		//error.print();
		//std::cout << std::endl << std::endl;
		x -= 3000;
	}


    //Matrix deltaError = MulMatrix(error,lr);
    Matrix deltaError = error;
    error.~Matrix();
	int nSamples = 1;
	Matrix weightChange = MulMatrix(deltaError, hiddenLayersActivations[nHiddenLayers - 1].transpose());
    Matrix biasChange = deltaError.reduce();

    weightChange = MulMatrix(weightChange, lr);
    biasChange = MulMatrix(biasChange, lr);

    for (int i = nHiddenLayers - 1; i > 0; i--) {
        if (i == nHiddenLayers - 1) {
            deltaError = MulMatrix(outputWeights.transpose(), deltaError);
			deltaError = HadamardProduct(deltaError, applyFunction(hiddenLayersActivations[i], hiddenLayers[i]->activationDerivative));
            outputWeights = SubMatrix(outputWeights, weightChange);
			outputWeights = MulMatrix(outputWeights, 1.0 / nSamples);
            outputBiases = SubMatrix(outputBiases, biasChange);
        }
        else {
            deltaError = MulMatrix(hiddenLayers[i + 1]->getWeights().transpose(), deltaError);
            deltaError = HadamardProduct(deltaError, applyFunction(hiddenLayersActivations[i], hiddenLayers[i]->activationDerivative));
            hiddenLayers[i + 1]->updateWeights(weightChange, nSamples);
            hiddenLayers[i + 1]->updateBiases(biasChange);
        }
        weightChange.~Matrix();
        biasChange.~Matrix();

        weightChange = MulMatrix(deltaError, hiddenLayersActivations[i - 1].transpose());

        biasChange = deltaError.reduce();
        weightChange = MulMatrix(weightChange, lr);
        biasChange = MulMatrix(biasChange, lr);
        
    }

	// Calculate the error for the first hidden layer
	deltaError = MulMatrix(hiddenLayers[1]->getWeights().transpose(), deltaError);
	deltaError = HadamardProduct(deltaError, applyFunction(hiddenLayersActivations[0], hiddenLayers[0]->activationDerivative));
    
	// Update the second layer weights and biases
    hiddenLayers[1]->updateWeights(weightChange, nSamples);
    hiddenLayers[1]->updateBiases(biasChange);

    weightChange.~Matrix();
    biasChange.~Matrix();

    //weightChange = (deltaError * inputs) * lr;
    weightChange = MulMatrix(deltaError, inputs.transpose());
	weightChange = MulMatrix(weightChange, lr);
	biasChange = MulMatrix(deltaError.reduce(), lr);
	//biasChange = deltaError;
    hiddenLayers[0]->updateWeights(weightChange, nSamples);
    hiddenLayers[0]->updateBiases(biasChange);
    weightChange.~Matrix();
    biasChange.~Matrix();

    deltaError.~Matrix();
}
