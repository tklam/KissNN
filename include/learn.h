#ifndef _KISS_NN_LEARN_NN_H_
#define _KISS_NN_LEARN_NN_H_

#include "KissNN.h"

namespace KissNN {

struct Sigmoid : public ActivationFunc {
    float operator() (std::vector<Edge*> & inputEdges) const;
    float derivative (Edge * edge) const override;
};

struct Constant: public ActivationFunc {
    float _value;
    float operator() (std::vector<Edge*> & inputEdges) const override;
    float derivative (Edge * edge) const override;
};

struct SquaredError: public Criterion {
    float operator() (OutputNode* outputNode) const override;
    float derivative (OutputNode* outputNode) const override;
};

struct ForwardPass {
    virtual void operator() (Network* neuralNetwork) const;
};


/* simple stochastic gradient descent */
struct BackwardPropagation {
    virtual void operator() (Network* neuralNetwork) const;
};

/* simple stochastic weight update after each forward-backward pass */
struct UpdateWeights {
    float _learningRate;
    void operator() (Network* neuralNetwork) const;
};

}
#endif
