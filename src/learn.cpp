#include <stack>
#include <cmath>

#include "learn.h"

using namespace KissNN;
using namespace std;

//-------------------------------------------------------------------- learn
void ForwardPass::operator() (Network* neuralNetwork) const {
    // 1. reset visited flag
    const vector<Node*> allNodes = neuralNetwork->getAllNodes();
    for (auto && n:allNodes) {
        n->unsetFlag(KissNN::Node::VISITED);
    }

    // 2. for each output node, calc its value

    // from outputs to inputs
    stack<Node*> dfs;
    const vector<Node*> outputNodes = neuralNetwork->getOutputNodes();
    for (auto && n:outputNodes) {
        dfs.push(n);
    }
    
    while (dfs.empty() == false) {
        Node* n = dfs.top();

        bool isVisited = n->isFlagSet(KissNN::Node::VISITED);
        bool isInputNode = (n->_nodeType == KissNN::Node::INPUT);

        if (isVisited == false) {
            n->setFlag(KissNN::Node::VISITED);

            auto inputEdges = n->getInputEdges();
            for (auto &&e:inputEdges) {
                dfs.push(e->_source);
            }
        }
        
        if (   isInputNode 
            || isVisited
           ) {
            n->calcValue();
            dfs.pop();
        }
    }
}

void BackwardPropagation::operator() (Network* neuralNetwork) const {

    // 1. reset visited flag
    const vector<Node*> allNodes = neuralNetwork->getAllNodes();
    for (auto && n:allNodes) {
        n->unsetFlag(KissNN::Node::VISITED);
    }

    // 2. for each edge, calc its grad weight

    // from inputs to outputs
    stack<Edge*> dfs;
    const vector<Node*> inputNodes = neuralNetwork->getInputNodes();
    for (auto && n:inputNodes) {
        auto outputEdges = n->getOutputEdges();
        for (auto &&e:outputEdges) {
            dfs.push(e);
        }
    }
    
    while (dfs.empty() == false) {
        Edge* edge = dfs.top();
        Node * n = edge->_destination;

        bool isDstVisited = n->isFlagSet(KissNN::Node::VISITED);
        bool isDstOutputNode = (n->_nodeType == KissNN::Node::OUTPUT);

        if (isDstVisited == false) {
            n->setFlag(KissNN::Node::VISITED);

            auto outputEdges = n->getOutputEdges();
            for (auto &&e:outputEdges) {
                dfs.push(e);
            }
        }
        
        if (   isDstOutputNode 
            || isDstVisited
           ) {
            edge->calcGradWeight();
            dfs.pop();
        }
    }
}

void UpdateWeights::operator() (Network* neuralNetwork) const {
    const vector<Node*> allNodes = neuralNetwork->getAllNodes();

    for (auto && n:allNodes) {
        if (n->_nodeType == Node::OUTPUT) {
            continue;
        }

        auto outputEdges = n->getOutputEdges();
        for (auto && e:outputEdges) {
            e->_weight = e->_weight - _learningRate * e->_gradientWeight;
        }
    }
}

//-------------------------------------------------------------------- Sigmoid
float Sigmoid::operator() (std::vector<Edge*> & inputEdges) const {
    float x = 0;
    for (auto && e:inputEdges) {
        Node* source = e->_source;
        x += source->getValue() * e->_weight;
    }
    return 1/(1+pow(exp(1), -x));
}

float Sigmoid::derivative (Edge * edge) const {
    return _value * (1-_value);
}

//-------------------------------------------------------------------- Constant
float Constant::operator() (std::vector<Edge*> & inputEdges) const {
    return _value;
}

float Constant::derivative (Edge * edge) const {
    return 0;
}

//-------------------------------------------------------------------- SquaredError
float SquaredError::operator() (OutputNode* outputNode) const {
    return 0.5*pow((_targetValue - outputNode->getValue()), 2);
}

float SquaredError::derivative (OutputNode* outputNode) const {
    return (outputNode->getValue() - _targetValue);
}
