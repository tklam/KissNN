#include <iostream>

#include "learn.h"
#include "print.h"

#define NUM_TRAINING_SAMPLE (10)
#define NUM_TESTING_SAMPLE (10)

using namespace KissNN;
using namespace std;

int main (int argc, char** argv) {
    //srand(time(NULL));

    //manual config...
    //------------- output layer
    OutputNode output;
    output._name = "f";
    
    Sigmoid sigmoid;
    output._activationFunc = &sigmoid;

    SquaredError squaredError;
    output._criterion = &squaredError;

    //------------- interal layer 1
    InternalNode internal_1, internal_2;
    internal_1._name = "i1";
    internal_2._name = "i2";

    Sigmoid sigmoid_i1, sigmoid_i2;
    internal_1._activationFunc = &sigmoid_i1;
    internal_2._activationFunc = &sigmoid_i2;
   
    //------------- input layer
    InputNode input_1, input_2, input_3;
    input_1._name = "a";
    input_2._name = "b";
    input_3._name = "c";

    Constant input_1_value, input_2_value, input_3_value;
    input_1._activationFunc = &input_1_value;
    input_2._activationFunc = &input_2_value;
    input_3._activationFunc = &input_3_value;

    //------------- make connections
    output.addInput(&internal_1);
    output.addInput(&internal_2);

    internal_1.addInput(&input_1);
    internal_1.addInput(&input_2);
    internal_1.addInput(&input_3);

    internal_2.addInput(&input_1);
    internal_2.addInput(&input_2);
    internal_2.addInput(&input_3);

    Network network;
    network.addNode(&output);
    network.addNode(&input_1);
    network.addNode(&input_2);
    network.addNode(&input_3);
    network.addNode(&internal_1);
    network.addNode(&internal_2);

    //--- training    

    float a[NUM_TRAINING_SAMPLE] = {0.5, 0.3, 1, 0.25, 0.9, 0.5, 0.41, 0.6, 0.3, 0.7};
    float b[NUM_TRAINING_SAMPLE] = {0.3, 0.5, 0.2, 0.1, 0.8, 0.5, 0.4, 0.61, 0.31, 0.6};
    float targetValue[NUM_TRAINING_SAMPLE] = {1, 0, 1, 1, 1, 0, 1, 0, 0, 1};

    ForwardPass forwardPass;
    BackwardPropagation backPropagation;
    UpdateWeights updateWeights;
    updateWeights._learningRate = 0.75;

    for (size_t epoch=0; epoch<1000; ++epoch) {
        for (size_t i=0; i<NUM_TRAINING_SAMPLE; ++i) {
            input_1_value._value = a[i];
            input_2_value._value = b[i];
            input_3_value._value = -0.5;
            output._criterion->_targetValue = targetValue[i];

            forwardPass(&network);

            backPropagation(&network);

            updateWeights(&network);
        }
    }

    //--- testing
    
    float test_a[NUM_TESTING_SAMPLE] = {0.5, 0.3, 1, 0.2, 0.9, 0.5, 0.51, 0.61, 0.2, 0.4};
    float test_b[NUM_TESTING_SAMPLE] = {0.3, 0.5, 0.2, 1, 0.8, 0.5, 0.5, 0.60, 0.1, 0.5};
    float test_targetValue[NUM_TESTING_SAMPLE] = {1, 0, 1, 0, 1, 0, 1, 1, 1, 0};

    int numCorrect = 0;

    for (size_t i=0; i<NUM_TESTING_SAMPLE; ++i) {
        input_1_value._value = test_a[i];
        input_2_value._value = test_b[i];
        input_3_value._value = -0.5;
        output._criterion->_targetValue = test_targetValue[i];
        forwardPass(&network);        

        if (    output.getValue() > 0.5
            &&  static_cast<int>(test_targetValue[i]) == 1) {
            ++numCorrect;
        }
        else if (    output.getValue() <= 0.5
                 &&  static_cast<int>(test_targetValue[i]) == 0) {
            ++numCorrect;
        }
    }

    cout << "accuracy: " << (numCorrect / 10.0f) << endl;

    PrintNetwork printNetwork;
    printNetwork(&network);
}
