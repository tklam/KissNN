#include "KissNN.h"
#include <cstdlib>

using namespace KissNN;

//-------------------------------------------------------------------- Node
Node::Node():
    _name(""),
    _nodeType(Node::INTERNAL)
{
}
Node::~Node() {
    for (auto && e:_inputEdges) {
        delete e;
    }
}

Edge * Node::addInput(Node* input) {
    Edge * edge = new Edge();
    edge->_source = input;
    edge->_destination = this;
    edge->_weight = (rand() % 1001) / 1000.0f; //rand [0.0f, 1.0f] // 0;
    _inputEdges.push_back(edge);
    input->addOutputEdge(edge);
    return edge;
}

void Node::addOutputEdge(Edge* edge) {
    _outputEdges.push_back(edge);
}

void Node::calcValue() {
    _activationFunc->_value = (*_activationFunc)(_inputEdges);
}

float Node::getValue() const {
    return _activationFunc->_value;
}

const std::vector<Edge*> & Node::getInputEdges() const {
    return _inputEdges;
}

const std::vector<Edge*> & Node::getOutputEdges() const {
    return _outputEdges;
}

void Node::setFlag(Flag flag) {
    _flag = _flag | flag;
}

void Node::unsetFlag(Flag flag) {
    _flag = _flag & ~flag;
}

bool Node::isFlagSet(Flag flag) const {
    if ((_flag & flag) != 0) {
        return true;
    }
    return false;
}

Node::NodeType Node::getNodeType() const {
    return _nodeType;
}

//-------------------------------------------------------------------- Edge
Edge::Edge():
    _source(nullptr),
    _destination(nullptr)
{
}

Edge::~Edge() {
}

void Edge::calcGradWeight() {
    OutputNode* outputNode = static_cast<OutputNode*>(_destination);
    OutputNode* inputNode = static_cast<OutputNode*>(_source);

    switch (_destination->_nodeType) {
    case Node::OUTPUT:
        {
            _delta =   outputNode->_criterion->derivative(outputNode)
                     * outputNode->_activationFunc->derivative(this);
        }
        break;
    case Node::INTERNAL:
        {
            float sumOutputNodeGradWeights = 0;
            auto dstOutputEdges = _destination->getOutputEdges();
            for (auto && e:dstOutputEdges) {
                sumOutputNodeGradWeights += e->_weight * e->_delta;
            }
            _delta =   outputNode->_activationFunc->derivative(this)
                     * sumOutputNodeGradWeights;
        }
        break;
    default:
        break;
    }    

    _gradientWeight = _delta  * inputNode->getValue();
}

//-------------------------------------------------------------------- Network
Network::Network() {
}

Network::~Network() {
}

void Network::addNode(Node* node) {
    switch (node->_nodeType) {
    case Node::OUTPUT:
        _outputNodes.push_back(node);
        break;
    case Node::INPUT:
        _inputNodes.push_back(node);
        break;
    case Node::INTERNAL:
        _internalNodes.push_back(node);
        break;
    }
    _allNodes.push_back(node);
}

const std::vector<Node*> & Network::getInputNodes() const {
    return _inputNodes;
}

const std::vector<Node*> & Network::getInternalNodes() const {
    return _internalNodes;
}

const std::vector<Node*> & Network::getOutputNodes() const {
    return _outputNodes;
}

const std::vector<Node*> & Network::getAllNodes() const {
    return _allNodes;
}

//-------------------------------------------------------------------- InternalNode
InternalNode::InternalNode() {
}

InternalNode::~InternalNode() {
}

//-------------------------------------------------------------------- OutputNode
OutputNode::OutputNode() {
    _nodeType = Node::OUTPUT;
}

OutputNode::~OutputNode() {
}

//-------------------------------------------------------------------- InputNode
InputNode::InputNode() {
    _nodeType = Node::INPUT;
}

InputNode::~InputNode() {
}
