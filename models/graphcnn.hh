#ifndef GRAPHCNN_HH
#define GRAPHCNN_HH

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "linear.hh"
#include "batchnorm.hh"
#include "mlp.hh"

class GraphCNN {
private:
int num_layers_, mlp_num_layers_;
int input_dim_, hidden_dim_, output_dim_;
float drop_out_, learn_eps_;
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

public:
    GraphCNN(
        std::map<std::string, std::vector<std::vector<float>> > &data, 
        float drop_out, float learn_eps, 
        const std::string &graph_pooling_type, const std::string &neighbor_pooling_type
    );
    ~GraphCNN();
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
    float drop_out, float learn_eps, 
    const std::string &graph_pooling_type, const std::string &neighbor_pooling_type
) {
    num_layers_ = data["eps"][0].size() + 1;
    if (num_layers_ <= 1) {
        std::cerr << "error: invalid value of num_layer!" << std::endl;
        exit(0);
    }
    drop_out_ = drop_out;
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

#endif