#define DEBUG
#include "NeuralNetwork.h"
#include <iostream>
#include "Matrix.h"
#include "Matrix.h"
#include <algorithm>
#include "MNIST_Reader.h"

using namespace std;

int main()
{
	srand(time(0));
	vector<vector<double>> ar;
	ReadMNIST(10000, 784, ar);
	// normalize the data
	for (int i = 0; i < ar.size(); i++) {
		for (int j = 0; j < ar[i].size(); j++) {
			//ar[i][j] /= 255;
		}
	}
	// 75% train and 25% validation
	auto ThreeQuarters = ar.begin() + 3 * ar.size() / 4;
	vector<vector<double>> train(ar.begin(), ThreeQuarters);
	vector<vector<double>> validation(ThreeQuarters, ar.end());

	vector<int> labels;
	ReadMNIST_Label(10000, labels);
	auto ThreeQuartersLabels = labels.begin() + 3 * labels.size() / 4;
	vector<int> train_labels(labels.begin(), ThreeQuartersLabels);
	vector<int> validation_labels(ThreeQuartersLabels, labels.end());

	cout << "Training: " << train.size() << " " << train_labels.size() << endl;
	cout << "Validation: " << validation.size() << " " << validation_labels.size() << endl;



	NeuralNetwork nn(784, 10, 0.01);
	nn.addLayer(10, "ReLU");
	nn.addLayer(10, "ReLU");



	nn.Train(train, train_labels, 100, 64);

	nn.Validate(validation, validation_labels, 32);


	return 0;
}
/*
int main() {
	NeuralNetwork nn(2, 1, 0.3);

	nn.addLayer(2, "tanh");
	nn.addLayer(5, "tanh");

	double* test;
	test = new double[1];
	test[0] = 0.5;


	std::cout << "doubleest";



	Matrix* inputs = new Matrix[4];
	inputs[0] = Matrix(2, 1, false);
	inputs[1] = Matrix(2, 1, false);
	inputs[2] = Matrix(2, 1, false);
	inputs[3] = Matrix(2, 1, false);

	inputs[0][0][0] = 0;
	inputs[0][1][0] = 0;

	inputs[1][0][0] = 0;
	inputs[1][1][0] = 1;

	inputs[2][0][0] = 1;
	inputs[2][1][0] = 0;

	inputs[3][0][0] = 1;
	inputs[3][1][0] = 1;



	Matrix* outputs = new Matrix[4];

	outputs[0] = Matrix(1, 1, false);
	outputs[1] = Matrix(1, 1, false);
	outputs[2] = Matrix(1, 1, false);
	outputs[3] = Matrix(1, 1, false);

	outputs[0][0][0] = 0;
	outputs[1][0][0] = 1;
	outputs[2][0][0] = 1;
	outputs[3][0][0] = 0;

	int order[4] = { 0, 1, 2, 3 };

	for (int i = 0; i < 10000; i++) {
		int index = rand() % 4;
		nn.Train(inputs[order[index]], outputs[order[index]]);
	}
	


	return 0;
}
*/