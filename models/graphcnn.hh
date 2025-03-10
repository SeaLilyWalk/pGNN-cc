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
    void preprocess_neighbors_sumavepool(
        const std::vector<S2VGraph*> &data, MyMatrix *adj_block
    );
    MyMatrix* maxpool(const std::vector<S2VGraph*> &data,MyMatrix* h, int max_degree);
    MyMatrix* nextLayer(
        MyMatrix* h, const std::vector<S2VGraph*> &data, 
        int layer_idx, int max_degree, MyMatrix* neighbor_block
    );

public:
    GraphCNN(
        std::map<std::string, std::vector<std::vector<float>> > &data, 
        bool learn_eps, 
        const std::string &graph_pooling_type, const std::string &neighbor_pooling_type
    );
    ~GraphCNN();

    int get_input_dim();
    int get_output_dim();
    void forward(const std::vector<S2VGraph*> &data, int tag_sum, MyMatrix &output);
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
        mlp_data.push_back(data[bn_tag + std::to_string(i) + ".running_mean"][0]);
        mlp_data.push_back(data[bn_tag + std::to_string(i) + ".running_var"][0]);
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
        std::string rm_tag = "batch_norms." + std::to_string(i) + ".running_mean";
        std::string rv_tag = "batch_norms." + std::to_string(i) + ".running_var";
        batchnorms_.push_back(
            new BatchNorm(
                hidden_dim_, data[gamma_tag][0], data[beta_tag][0], 
                data[rm_tag][0], data[rv_tag][0]
            )
        );
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


inline int GraphCNN::get_input_dim() {
    return input_dim_;
}


inline int GraphCNN::get_output_dim() {
    return output_dim_;
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


MyMatrix* GraphCNN::maxpool(
    const std::vector<S2VGraph*> &data, MyMatrix* h, int max_degree
) {
    MyMatrix* pooled_rep = new MyMatrix(h->get_col_width(), h->get_row_width());
    std::vector<float> dummy;
    int row_length = h->get_row_width();
    for (int i = 0; i < row_length; ++i)
        dummy.push_back(h->get_min_val(0, i));
    int begin_idx = 0;
    for (const auto &g : data) {
        int g_node_sum = g->get_node_sum();
        auto neighbor_list = g->get_neighbors();
        for (int i = 0; i < g_node_sum; ++i) {
            std::vector<std::vector<float>> neighbors;
            if (neighbor_list[i].size() < max_degree)
                neighbors.push_back(dummy);
            if (!learn_eps_) {
                neighbors.push_back(std::vector<float>());
                h->get_row(i+begin_idx, neighbors.back());
            }
            for (auto n : neighbor_list[i]) {
                neighbors.push_back(std::vector<float>());
                h->get_row(i+begin_idx, neighbors.back());
            }
            int l = neighbors.size();
            for (int j = 0; j < row_length; ++j) {
                float max_val = neighbors[0][j];
                for (int k = 1; k < l; ++k)
                    max_val = std::max(max_val, neighbors[k][j]);
                pooled_rep->set_value(max_val, i+begin_idx, j);
            }
        }
        begin_idx += g_node_sum;
    }
    return pooled_rep;
}


MyMatrix* GraphCNN::nextLayer(
    MyMatrix* h, const std::vector<S2VGraph*> &data, 
    int layer_idx, int max_degree, MyMatrix* neighbor_block
) {
    MyMatrix *pooled;
    if (neighbor_pooling_type_ == "max") {
        pooled = maxpool(data, h, max_degree);
    } else {
        pooled = new MyMatrix(neighbor_block->get_col_width(), h->get_row_width());
        pooled->mult(*(neighbor_block), *(h));
        if (neighbor_pooling_type_ == "average") {
            for (int i = 0; i < neighbor_block->get_col_width(); ++i) {
                float degree_sum = 0;
                for (int j = 0; j < neighbor_block->get_row_width(); ++j)
                    degree_sum += neighbor_block->get_value(i, j);
                for (int j = 0; j < neighbor_block->get_row_width(); ++j) {
                    float tmp = neighbor_block->get_value(i, j);
                    tmp /= degree_sum;
                    neighbor_block->set_value(tmp, i, j);
                }
            } // for (int i=0)
        } // if (neighbor_pooling_type_ == "average")
    }
    if (learn_eps_) {
        MyMatrix tmp(h->get_col_width(), h->get_row_width());
        tmp.copy(*(h));
        tmp.mult(epss_[layer_idx] + 1);
        pooled->add(*(pooled), tmp);
    }
    MyMatrix *pooled_rep_t = new MyMatrix(hidden_dim_, pooled->get_col_width());
    MyMatrix *pooled_t = new MyMatrix(pooled->get_row_width(), pooled->get_col_width());
    pooled_t->transpose(*(pooled));
    mlps_[layer_idx]->forward(*(pooled_t), *(pooled_rep_t));
    batchnorms_[layer_idx]->forward(*(pooled_rep_t), *(pooled_rep_t));
    pooled_rep_t->activation(*(pooled_rep_t), "ReLU");
    MyMatrix *pooled_rep = new MyMatrix(pooled_rep_t->get_row_width(), pooled_rep_t->get_col_width());
    pooled_rep->transpose(*(pooled_rep_t));
    delete pooled;
    delete pooled_t;
    delete pooled_rep_t;
    return pooled_rep;
}


void GraphCNN::forward(
    const std::vector<S2VGraph*> &data, int tag_sum, MyMatrix &output
) {
    // get node features
    int node_sum = 0;
    for (const auto &g : data)
        node_sum += g->get_node_sum();
    MyMatrix *node_feature = new MyMatrix(node_sum, tag_sum);
    get_node_feature(data, *(node_feature));

    // get graph pool
    MyMatrix graph_pool(data.size(), node_sum);
    preprocess_graphpool(data, graph_pool);

    MyMatrix *neighbor_block;
    int max_deg = 0;
    for (const auto &g : data)
        max_deg = std::max(g->get_max_degree(), max_deg);

    // get neibor list
    if (neighbor_pooling_type_ != "max") {
        neighbor_block = new MyMatrix(node_sum, node_sum);
        preprocess_neighbors_sumavepool(data, neighbor_block);
    }

    std::vector<MyMatrix*> hidden_rep;
    hidden_rep.push_back(node_feature);

    for (int layer_idx = 0; layer_idx < num_layers_-1; ++layer_idx) 
        hidden_rep.push_back(
            nextLayer(hidden_rep.back(), data, layer_idx, max_deg, neighbor_block)
        );
    
    int row_size;
    for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
        if (layer_idx == 0)
            row_size = input_dim_;
        else
            row_size = hidden_dim_;
        MyMatrix pooled_h(data.size(), row_size);
        pooled_h.mult(graph_pool, *(hidden_rep[layer_idx]));
        delete hidden_rep[layer_idx];
        MyMatrix tmp(output_dim_, data.size());
        MyMatrix pooled_h_t(row_size, data.size());
        pooled_h_t.transpose(pooled_h);
        linears_[layer_idx]->forward(pooled_h_t, tmp);
        output.add(output, tmp);
    }

    if (neighbor_pooling_type_ != "max")
        delete neighbor_block;
}

#endif