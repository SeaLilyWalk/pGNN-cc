#ifndef GRAPHCNN_HH
#define GRAPHCNN_HH

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "linear.hh"
#include "batchnorm.hh"
#include "mlp.hh"
#include "../s2vgraph.hh"

class GraphCNN {
private:
    int num_layers_, mlp_num_layers_;
    int input_dim_, hidden_dim_, output_dim_;
    bool learn_eps_;
    std::string graph_pooling_type_, neighbor_pooling_type_;
    std::vector<float> epss_;
    std::vector<Linear*> linears_;
    std::vector<BatchNorm*> batchnorms_;
    std::vector<MLP*> mlps_;

    void build_linear(
        const std::string& tag, 
        std::map<std::string, std::vector<std::vector<float>> > &data
    );
    void build_mlp(
        const std::string& tag, int input_dim, 
        std::map<std::string, std::vector<std::vector<float>> > &data
    );

    void get_node_feature(const std::vector<S2VGraph*> &data, MyMatrix& node_feature);
    void preprocess_graphpool(const std::vector<S2VGraph*> &data, MyMatrix& graph_pool);
    void preprocess_neighbors_maxpool(
        const std::vector<S2VGraph*> &data, MyMatrix *padded_neighbor_list
    );
    void preprocess_neighbors_sumavepool(
        const std::vector<S2VGraph*> &data, MyMatrix *adj_block
    );

public:
    GraphCNN(
        std::map<std::string, std::vector<std::vector<float>> > &data, 
        bool learn_eps, 
        const std::string &graph_pooling_type, const std::string &neighbor_pooling_type
    );
    ~GraphCNN();

    void forward(const std::vector<S2VGraph*> &data, int tag_sum, std::vector<int> &output);
};


void GraphCNN::build_linear(
    const std::string& tag, 
    std::map<std::string, std::vector<std::vector<float>> > &data
) {
    std::string weight_tag = tag + ".weight";
    std::string bias_tag = tag + ".bias";
    int input_dim = data[weight_tag][0].size();
    int output_dim = data[weight_tag].size();
    linears_.push_back(
        new Linear(input_dim, output_dim, data[weight_tag], data[bias_tag][0])
    );
}


// tag: mlps.x.
void GraphCNN::build_mlp(
    const std::string& tag, int input_dim,
    std::map<std::string, std::vector<std::vector<float>> > &data
) {
    std::string linear_tag = tag + "linears.";
    std::vector<std::vector<float>> mlp_data;
    for (int i = 0; i < mlp_num_layers_; ++i) {
        std::string weight_tag = linear_tag + std::to_string(i) + ".weight";
        std::string bias_tag = linear_tag + std::to_string(i) + ".bias";
        for (const auto &line : data[weight_tag])
            mlp_data.push_back(line);
        mlp_data.push_back(data[bias_tag][0]);
    }
    std::string bn_tag = tag + "batch_norms.";
    for (int i = 0; i < mlp_num_layers_-1; ++i) {
        mlp_data.push_back(data[bn_tag + std::to_string(i) + ".weight"][0]);
        mlp_data.push_back(data[bn_tag + std::to_string(i) + ".bias"][0]);
    }
    mlps_.push_back(new MLP(input_dim, hidden_dim_, hidden_dim_, mlp_num_layers_, mlp_data));
}


GraphCNN::GraphCNN(
    std::map<std::string, std::vector<std::vector<float>> > &data, 
    bool learn_eps, 
    const std::string &graph_pooling_type, const std::string &neighbor_pooling_type
) {
    num_layers_ = data["eps"][0].size() + 1;
    if (num_layers_ <= 1) {
        std::cerr << "error: invalid value of num_layer!" << std::endl;
        exit(0);
    }
    learn_eps_ = learn_eps;
    graph_pooling_type_ = graph_pooling_type;
    neighbor_pooling_type_ = neighbor_pooling_type;
    for (auto e : data["eps"][0]) 
        epss_.push_back(e);
    // linear, and get the size of model by the way
    input_dim_ = data["linears_prediction.0.weight"][0].size();
    output_dim_ = data["linears_prediction.0.weight"].size();
    build_linear("linears_prediction.0", data);
    hidden_dim_ = data["linears_prediction.1.weight"][0].size();
    std::string linear_tag = "linears_prediction.";
    for (int i = 1; i < num_layers_; ++i)
        build_linear(linear_tag+std::to_string(i), data);
    // batchnorm
    for (int i = 0; i < num_layers_-1; ++i) {
        std::string gamma_tag = "batch_norms." + std::to_string(i) + ".weight";
        std::string beta_tag = "batch_norms." + std::to_string(i) + ".bias";
        batchnorms_.push_back(new BatchNorm(hidden_dim_, data[gamma_tag][0], data[beta_tag][0]));
    }
    // mlp
    mlp_num_layers_ = 0;
    while (true) {
        if (data.find("mlps.0.linears." + std::to_string(mlp_num_layers_) + ".bias") == data.end())
            break;
        ++mlp_num_layers_;
    }
    build_mlp("mlps.0.", input_dim_, data);
    for (int i = 1; i < num_layers_-1; ++i)
        build_mlp("mlps." + std::to_string(i) + ".", hidden_dim_, data);
}

GraphCNN::~GraphCNN() {
    for (auto p : linears_)
        delete p;
    for (auto p : batchnorms_)
        delete p;
    for (auto p : mlps_)
        delete p;
}


void GraphCNN::get_node_feature(
    const std::vector<S2VGraph*> &data, MyMatrix &node_feature
) {
    int begin_idx = 0;
    for (const auto &g : data) {
        auto f = g->get_node_features();
        for (const auto &p : f)
            node_feature.set_value(1, p.first + begin_idx, p.second);
        begin_idx += g->get_node_sum();
    }
}


void GraphCNN::preprocess_graphpool(
    const std::vector<S2VGraph*> &data, MyMatrix& graph_pool
) {
    int begin_idx = 0;
    for (int i = 0; i < data.size(); ++i) {
        float elem = 0;
        int g_node_sum = data[i]->get_node_sum();
        if (graph_pooling_type_ == "average")
            elem = 1/float(g_node_sum);
        else 
            elem = 1;
        for (int j = 0; j < g_node_sum; ++j)
            graph_pool.set_value(elem, i, begin_idx+j);
        begin_idx += g_node_sum;
    }
}


void GraphCNN::preprocess_neighbors_maxpool(
    const std::vector<S2VGraph*> &data, MyMatrix *padded_neighbor_list
) {
    int max_degree = padded_neighbor_list->get_row_width();
    if (!learn_eps_)
        max_degree--;
    int begin_idx = 0;
    for (int i = 0; i < data.size(); ++i) {
        auto g_neighbors = data[i]->get_neighbors();
        int g_node_sum = data[i]->get_node_sum();
        for (int j = 0; j < g_node_sum; ++j) {
            int cnt = 0;
            for (auto k : g_neighbors[j]) 
                padded_neighbor_list->set_value(k+begin_idx, j+begin_idx, cnt++);
            for (int k = g_neighbors[j].size(); k < max_degree; ++k)
                padded_neighbor_list->set_value(-1, j+begin_idx, k);
            if (!learn_eps_)
                padded_neighbor_list->set_value(j+begin_idx, j+begin_idx, max_degree);
        }
        begin_idx += data[i]->get_node_sum();
    }
}


void GraphCNN::preprocess_neighbors_sumavepool(
    const std::vector<S2VGraph*> &data, MyMatrix *adj_block
) {
    int begin_idx = 0;
    for (auto g : data) {
        int g_node_sum = g->get_node_sum();
        auto g_edges = g->get_edges();
        for (const auto &p : g_edges)
            adj_block->set_value(1, p.first+begin_idx, p.second+begin_idx);
        if (!learn_eps_) {
            for (int i = 0; i < g_node_sum; ++i)
                adj_block->set_value(1, i+begin_idx, i+begin_idx);
        }
        begin_idx += g_node_sum;
    }
}


void GraphCNN::forward(
    const std::vector<S2VGraph*> &data, int tag_sum, std::vector<int> &output
) {
    // get node features
    int node_sum = 0;
    for (const auto &g : data)
        node_sum += g->get_node_sum();
    MyMatrix node_feature(node_sum, tag_sum);
    get_node_feature(data, node_feature);

    // get graph pool
    MyMatrix graph_pool(data.size(), node_sum);
    preprocess_graphpool(data, graph_pool);

    MyMatrix *neighbor_block;

    // get neibor list
    if (neighbor_pooling_type_ == "max") {
        int max_deg = 0;
        for (const auto &g : data)
            max_deg = std::max(g->get_max_degree(), max_deg);
        int pad_size = max_deg;
        if (!learn_eps_)
            pad_size += 1;
        MyMatrix padded_neighbor_list(node_sum, pad_size);
        neighbor_block = new MyMatrix(node_sum, pad_size);
        preprocess_neighbors_maxpool(data, neighbor_block);
    } else {
        neighbor_block = new MyMatrix(node_sum, node_sum);
        preprocess_neighbors_sumavepool(data, neighbor_block);
    }

    std::vector<MyMatrix> hidden_rep;
    hidden_rep.push_back(node_feature);

    delete neighbor_block;
}

#endif