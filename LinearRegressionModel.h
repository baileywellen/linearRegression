#include <vector>
#include <iostream>
using namespace std;

#pragma once

vector<double> multiply_vectors(vector<double> features, vector<double> weights)
{
	if (features.size() != weights.size())
	{
		cout << "There is a problem" << endl;
	}
	vector <double> output;

	for (size_t i = 0; i < features.size(); i++)
	{
		double n = features.at(i) * weights.at(i);
		output.push_back(n);
	}

	return output;
}

double sum_vector(vector<double> numbers)
{
	double a = 0;
	for (size_t i = 0; i < numbers.size(); i++)
	{
		a = a + numbers.at(i);
	}

	return a;
}

class LinearRegressionModel
{
public:
	LinearRegressionModel(vector <vector<double>>& features, vector<double>& labels, double learningRate);
	double predict(vector<double>& inputFeatures);
	double calculate_yprime(vector<double> features);

private:
	vector<double> weights;
	vector<double> last_weights;

};

double LinearRegressionModel::predict(vector<double>& inputFeatures)
{
	//start our prediction at the y intercept
	double prediction = weights.at(0);

	//then, add the slope(weight) * each of the other features
	for (size_t i = 1; i < inputFeatures.size(); i++)
	{
		//weights is off by 1 from the features vector because of the y intercept 
		prediction += weights.at(i + 1) * inputFeatures.at(i);
	}
	return prediction;
}

//{ {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0} } -- the first column is {1,2,3} and the second is {1,2,3}
LinearRegressionModel::LinearRegressionModel(vector<vector<double>>& features, vector<double>& labels, double learningRate)
{
	for (size_t i = 0; i < features.size(); i++)
	{
		//add 1 for the y intercept in each vector
		features.at(i).insert(features.at(i).begin(), 1);
	}

	//initialize all weights to 0 - we should have 1 weight per column, which can be found by looking at the size of the inner vector
	for (size_t i = 0; i < features.at(0).size(); i++)
	{
		weights.push_back(0);
		last_weights.push_back(1);
	}

	//use this variable to see if we need to keep going
	bool keep_calculating = true;

	while (keep_calculating)
	{
		//calculate y' - y
		vector<double> differences;
		for (size_t i = 0; i < features.size(); i++)
		{
			double x = 0;
			for (size_t j = 0; j < weights.size(); j++)
			{
				x += features[i][j] * weights[j];
			}
			// push (yprime - y) back
			differences.push_back(x - labels[i]);
		}

		//calculate (y' - y) * xi / number of training points
		vector<double> diff_summed;
		//go through columns
		for (size_t i = 0; i < features.size(); i++)
		{
			double sum_col = 0;
			for (size_t j = 0; j < differences.size(); j++)
			{
				double multiplied = differences[j] * features[j][i];
				sum_col += multiplied;
			}

			sum_col = sum_col / differences.size();
			diff_summed.push_back(sum_col);
		}

		vector<double> calculated_weights;
		//calculate new weights
		for (size_t i = 0; i < diff_summed.size(); i++)
		{
			double temp = weights.at(i) - (diff_summed.at(i) * learningRate);
			calculated_weights.push_back(temp);
		}

		//change weights to be new values - save last values to check
		last_weights = weights;
		weights = calculated_weights;

		//check to see if we need to keep calculating
		bool all_close_enough = true;
		for (size_t i = 0; i < weights.size(); i++)
		{
			//if even one of the weights is more than 0.0001 off from the last one, we know that we have to keep going
			if (abs(weights.at(i) - last_weights.at(i)) > 0.0001)
			{
				all_close_enough = false;
			}
		}

		if (all_close_enough == true)
		{
			keep_calculating = false;
		}
	}	
}
