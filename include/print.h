#ifndef _KISS_NN_PRINT_NN_H_
#define _KISS_NN_PRINT_NN_H_

#include "KissNN.h"
#include <string>

namespace KissNN {

struct PrintNetwork {
    virtual void operator() (Network* neuralNetwork) const;
};

struct PrintEdge{
    virtual void operator() (Edge* edge, std::string indent) const;
};

struct PrintNode {
    PrintEdge * _printEdge;
    virtual void print(Node* node, std::string indent) const;
    virtual void operator() (InputNode* node, std::string indent) const;
    virtual void operator() (InternalNode* node, std::string indent) const;
    virtual void operator() (OutputNode* node, std::string indent) const;
};

}
#endif
