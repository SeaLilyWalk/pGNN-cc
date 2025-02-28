#ifndef S2VGRAPH_HH
#define S2VGRAPH_HH

#include <iostream>
#include <vector>
#include <set>

#include "models/my_matrix.hh"


class S2VGraph {
private:
    int num_nodes_;
    std::vector<std::set<int>> neighbors_;
    MyMatrix* edge_mat_;
    int label_;
    std::vector<int> node_tags_;
    MyMatrix* node_features_;

public:
    S2VGraph(int label, int num_nodes);
    ~S2VGraph();
    int get_node_sum();

    friend void loadData(
        const std::string& dataset, bool degree_as_tag, 
        std::vector<S2VGraph*> &graph_list, int &label_sum
    );
};


S2VGraph::S2VGraph(int label, int num_nodes) {
    num_nodes_ = num_nodes;
    for (int i = 0; i < num_nodes; ++i)
        neighbors_.push_back(std::set<int>());
    label_ = label;
}


S2VGraph::~S2VGraph() {
    delete edge_mat_;
    delete node_features_;
}


inline int S2VGraph::get_node_sum() {
    return num_nodes_;
}

#endif