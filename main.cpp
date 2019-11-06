#include <iostream>
#include <vector>
#include "LinearRegressionModel.h"
using namespace std;

int main()
{

	vector<vector<double>> features = { {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0} };
	vector<double> labels = { 3.0, 2.0, 1.0 };

	LinearRegressionModel model(features, labels, .1);

	vector<double> unknownInput = { 0.0, 4.0 };

	double predictedValue = model.predict(unknownInput); //should be about 2.0

	cout << "The predicted value is " << predictedValue << endl;
	system("pause");
	return 0;
}
