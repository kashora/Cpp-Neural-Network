#pragma once

#include <fstream>
#include <iostream>
#include <vector>


/// <summary>
///     Code to read the MNIST dataset
///     https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set/
/// 
/// </summary>

int ReverseInt(int i);

void ReadMNIST(int NumberOfImages, int DataOfAnImage, std::vector<std::vector<double>>& arr);

void ReadMNIST_Label(int NumberOfImages, std::vector<int>& arr);