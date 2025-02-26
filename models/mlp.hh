#ifndef MLP_HH
#define MLP_HH

#include <iostream>
#include <vector>

#include "my_matrix.hh"

class MLP {
private:
    int num_layers_;
    int input_dim_, hidden_dim_, output_dim_;
    std::vector<Linear*> linears_;
    std::vector<BatchNorm*> batchnorms_;

    void add_new_linear(
        int input_dim, int output_dim, int begin_idx,
        const std::vector<std::vector<float>> &model_data
    );
    void add_new_bn(
        int input_dim, int begin_idx,
        const std::vector<std::vector<float>> &model_data
    );

public:
    MLP(
        int input_dim, int hidden_dim, int output_dim, int num_layers, 
        const std::vector<std::vector<float>>& model_data
    );
    ~MLP();
    void forward(const MyMatrix& input, MyMatrix& output);
};


void MLP::add_new_linear(
    int input_dim, int output_dim, int begin_idx,
    const std::vector<std::vector<float>> &model_data
) {
    std::vector<float> bdata;
    std::vector<std::vector<float>> wdata;
    for (int i = begin_idx; i < begin_idx+output_dim; ++i)
        wdata.push_back(model_data[i]);
    for (auto num : model_data[begin_idx+output_dim])
        bdata.push_back(num);
    linears_.push_back(new Linear(input_dim, output_dim, wdata, bdata));
}


void MLP::add_new_bn(
    int input_dim, int begin_idx,
    const std::vector<std::vector<float>> &model_data
) {
    std::vector<float> gdata, bdata;
    for (auto num : model_data[begin_idx])
        gdata.push_back(num);
    for (auto num : model_data[begin_idx+1])
        bdata.push_back(num);
    batchnorms_.push_back(new BatchNorm(input_dim, gdata, bdata));
}


MLP::MLP(
    int input_dim, int hidden_dim, int output_dim, int num_layers, 
    const std::vector<std::vector<float>>& model_data
) {
    num_layers_ = num_layers;
    input_dim_ = input_dim;
    hidden_dim_ = hidden_dim;
    output_dim_ = output_dim;
    if (num_layers < 1) {
        std::cerr << "error: wrong size of mlp!" << std::endl;
        exit(0);
    }
    if (num_layers == 1) {
        add_new_linear(input_dim, output_dim, 0, model_data);
    } else {
        add_new_linear(input_dim, hidden_dim, 0, model_data);
        int begin_idx = hidden_dim+1;
        for (int i = 0; i < num_layers-2; ++i) {
            add_new_linear(hidden_dim, hidden_dim, begin_idx, model_data);
            begin_idx += hidden_dim+1;
        }
        add_new_linear(hidden_dim, output_dim, begin_idx, model_data);
        begin_idx += output_dim+1;
        for (int i = 0; i < num_layers-1; ++i) {
            add_new_bn(hidden_dim, begin_idx, model_data);
            begin_idx += 2;
        }
    }
}


MLP::~MLP() {
    for (auto p : linears_)
        delete p;
    for (auto p : batchnorms_)
        delete p;
}

#endif