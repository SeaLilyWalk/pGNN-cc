#ifndef UTIL_HH
#define UTIL_HH

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <random>

#include "s2vgraph.hh"


void loadData(
    const std::string& dataset, bool degree_as_tag, 
    std::vector<S2VGraph*> &graph_list, int &label_sum, int &tag_sum
) {
    std::cout << "Loading data..." << std::endl;

    // read the data in the file
    std::string path = "dataset/" + dataset + "/" + dataset + ".txt";
    std::ifstream data_in(path);
    std::string str;
    std::getline(data_in, str);
    int graphs_num = std::stoi(str); // the number of subgraphs
    std::string sub_inform;
    int g_label, n, n_edges;
    std::map<int, int> label_dict, feat_dict;
    for (int i = 0; i < graphs_num; ++i) {
        std::getline(data_in, sub_inform);
        std::istringstream data_info(sub_inform);
        data_info >> n >> g_label;
        if (label_dict.find(g_label) == label_dict.end())
            label_dict[g_label] = label_dict.size();
        n_edges = 0;
        S2VGraph* g = new S2VGraph(g_label, n);
        // get the information of each node in the subgraph
        for (int j = 0; j < n; ++j) {
            std::vector<int> row;
            std::getline(data_in, str);
            std::istringstream row_in(str);
            int m;
            while (row_in >> m)
                row.push_back(m);
            int tmp = row[1] + 2;
            if (feat_dict.find(row[0]) == feat_dict.end())
                feat_dict[row[0]] = feat_dict.size();
            g->node_tags_.push_back(feat_dict[row[0]]);
            n_edges += row[1];
            for (int k = 2; k < tmp; ++k) {
                g->neighbors_[j].insert(row[k]);
                g->neighbors_[row[k]].insert(j);
            }
        }
        graph_list.push_back(g);
    }
    data_in.close();

    // build the edge transpose matrix
    int max_degree = 0;
    std::set<int> tagset;
    for (auto& g : graph_list) {
        for (int i = 0; i < g->neighbors_.size(); ++i) {
            for (auto j : g->neighbors_[i]) 
                g->edges_.push_back(std::pair<int, int>(i, j));
            // get the max degree of the node in a subgraph
            max_degree = std::max(max_degree, int(g->neighbors_[i].size()));
        }
        g->max_degree_ = max_degree;
        // one-hot code the label of a graph
        g->label_ = label_dict[g->label_];
        if (degree_as_tag) {
            for (int i = 0; i < g->num_nodes_; ++i)
                g->node_tags_[i] = g->neighbors_[i].size();
        }
        // get the whole tag for one-hot code
        for (auto i : g->node_tags_)
            tagset.insert(i);
    }
    label_sum = label_dict.size();

    // build the node_features of the graph
    std::map<int, int> tag2idx;
    int cnt = 0;
    tag_sum = tagset.size();
    for (auto i : tagset) 
        tag2idx[i] = cnt++;
    for (auto &g : graph_list) {
        // g->node_features_ = new MyMatrix(g->node_tags_.size(), tagset.size());
        for (int i = 0; i < g->node_tags_.size(); ++i)
            g->node_features_.push_back(std::pair<int, int>(i, tag2idx[g->node_tags_[i]]));
    }

    std::cout << "# classes: " << label_dict.size() << std::endl;
    std::cout << "# maximum node tag: " << tagset.size() << std::endl;
    std::cout << "# graphs number: " << graph_list.size() << std::endl;
}


// according to k-fold cross validation
// choose random data for test_graph_list
void separateData(
    std::vector<S2VGraph*>& graph_list, int fold_idx,
    std::vector<S2VGraph*>& train_list, std::vector<S2VGraph*>& test_list
) {
    if (fold_idx < 0 || fold_idx >= 10) {
        std::cerr << "error: fold_idx must be from 0 to 9!" << std::endl;
        exit(0);
    }
    std::random_device seed;
	std::ranlux48 engine(seed());
    int l = graph_list.size();
    std::uniform_int_distribution<> distrib(0, l);
    int random = distrib(engine);
    int batch_size = l/fold_idx;
    std::set<int> idx_test;
    for (int i = 0; i < batch_size; ++i) {
        int idx = (i+random)%l;
        idx_test.insert(idx);
        test_list.push_back(graph_list[idx]);
    }
    for (int i = 0; i < l; ++i)
        if (idx_test.find(i) == idx_test.end())
            train_list.push_back(graph_list[i]);
}

#endif