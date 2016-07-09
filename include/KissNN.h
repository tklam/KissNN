#ifndef _KISS_NN_H_
#define _KISS_NN_H_

#include <vector>
#include <string>

namespace KissNN {

class Edge;
class Node;
class InputNode;
class InternalNode;
class OutputNode;

struct ActivationFunc {
    float _value;
    virtual float operator() (std::vector<Edge*> & inputEdges) const = 0;
    virtual float derivative (Edge * edge) const = 0;
};

struct Criterion {    
    float _targetValue;
    float _value;
    virtual float operator() (OutputNode* outputNode) const = 0;
    virtual float derivative (OutputNode* outputNode) const = 0;
};

class Node {    
private:
    std::vector<Edge*> _inputEdges;
    std::vector<Edge*> _outputEdges;
    unsigned int _flag;
public:
    enum Flag : unsigned int {
        VISITED = 1
    };
    enum NodeType {
        INPUT,
        INTERNAL,
        OUTPUT
    };

    Node();
    virtual ~Node() = 0;

    std::string _name;
    float getValue() const;
    NodeType _nodeType;
    ActivationFunc * _activationFunc;
    Edge * addInput(Node* input);
    void addOutputEdge(Edge* edge);
    void calcValue();
    const std::vector<Edge*> & getInputEdges() const;
    const std::vector<Edge*> & getOutputEdges() const;
    void setFlag(Flag flag);
    void unsetFlag(Flag flag);
    bool isFlagSet(Flag flag) const;
    NodeType getNodeType() const;
};

class Edge {
public:
    Node * _source;
    Node * _destination;
    float _weight;
    float _gradientWeight;
    float _delta; // _gradientWeight = _delta * _source->getValue()

    Edge();
    ~Edge();

    void calcGradWeight();
};

class Network {
private:
    std::vector<Node*> _inputNodes;
    std::vector<Node*> _internalNodes;
    std::vector<Node*> _outputNodes;
    std::vector<Node*> _allNodes;
public:
    Network();
    ~Network();
    void addNode(Node* node);
    const std::vector<Node*> & getInputNodes() const;
    const std::vector<Node*> & getInternalNodes() const;
    const std::vector<Node*> & getOutputNodes() const;
    const std::vector<Node*> & getAllNodes() const;
};

class InternalNode : public Node {
public:
    InternalNode();
    ~InternalNode() override;
};

class OutputNode: public Node {
public:
    OutputNode();
    ~OutputNode() override;
    Criterion * _criterion;
};

class InputNode : public Node {
public:
    InputNode();
    ~InputNode() override;
};

};

#endif
