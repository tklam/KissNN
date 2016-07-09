#include <iostream>

#include "print.h"

using namespace KissNN;
using namespace std;

void PrintNetwork::operator() (Network* neuralNetwork) const {
    const vector<Node*> outputNodes = neuralNetwork->getOutputNodes();
    const vector<Node*> inputNodes = neuralNetwork->getInputNodes();
    const vector<Node*> internalNodes = neuralNetwork->getInternalNodes();

    PrintNode printNode;
    PrintEdge printEdge;
    printNode._printEdge = &printEdge;

    for (auto && n:inputNodes) {
        printNode(static_cast<InputNode*>(n), "");
        cout << endl;
    }
    for (auto && n:internalNodes) {
        printNode(static_cast<InternalNode*>(n), "");
        cout << endl;
    }
    for (auto && n:outputNodes) {
        printNode(static_cast<OutputNode*>(n), "");
        cout << endl;
    }
}

void PrintNode::print(Node* node, std::string indent) const {
    cout << indent << "node: " << hex << node;
    if (node->_name != "") {
        cout << " name: " << node->_name;
    }
}


void PrintNode::operator() (InputNode* node, std::string indent) const {
    string curIndent = "    ";
    print(node, indent);
    cout << endl;
    cout << indent << curIndent << "- output edges: " << endl;
    auto outputEdges = node->getOutputEdges();
    for (auto && e:outputEdges) {
        (*_printEdge)(e, indent+curIndent+curIndent);
        cout << endl;
    }
    cout << indent << "- input value: " << node->getValue();
}

void PrintNode::operator() (InternalNode* node, std::string indent) const {
    string curIndent = "    ";
    print(node, indent);
    cout << endl;
    cout << indent << curIndent << "- output edges: " << endl;
    auto outputEdges = node->getOutputEdges();
    for (auto && e:outputEdges) {
        (*_printEdge)(e, indent+curIndent+curIndent);
        cout << endl;
    }
    cout << indent << "- value: " << node->getValue();
}

void PrintNode::operator() (OutputNode * node, std::string indent) const {
    string curIndent = "    ";
    print(node, indent);
    cout << endl;
    cout << indent << curIndent << "- value: " << node->getValue();
}

void PrintEdge::operator() (Edge* edge, string indent) const {
    string curIndent = "    ";
    cout << indent << "- " << hex << edge->_source << " -> " << hex << edge->_destination << endl;
    cout << indent << curIndent << "- weight: " << edge->_weight << endl;
    cout << indent << curIndent << "- gradient weight: " << edge->_gradientWeight << endl;
    cout << indent << curIndent << "- delta: " << edge->_delta;
}
