#ifndef LINEAR_HH
#define LINEAR_HH

#include <iostream>
#include <vector>

#include "my_matrix.hh"

class Linear {
private:
    MyMatrix* weight_;
    MyMatrix* bia_;
public:
    Linear(
        int input_dim, int output_dim, 
        const std::vector<std::vector<float>> &wdata, 
        const std::vector<float> &bdata
    );
    ~Linear();
    void forward(const MyMatrix& input, MyMatrix& output);
};


Linear::Linear(
    int input_dim, int output_dim, 
    const std::vector<std::vector<float>> &wdata, 
    const std::vector<float> &bdata
) {
    weight_ = new MyMatrix(output_dim, input_dim);
    std::vector<float> m;
    for (int i = 0; i < output_dim; ++i)
        for (auto num : wdata[i])
            m.push_back(num);
    weight_->copy(m);
    bia_ = new MyMatrix(output_dim, 1);
    bia_->copy(bdata);
}


void Linear::forward(const MyMatrix& input, MyMatrix& output) {
    output.mult(*(weight_), input);
    for (int i = 0; i < output.row_width_; ++i) {
        for (int j = 0; j < output.col_width_; ++j)
            output.mat_[j][i] += bia_->mat_[j][0];
    }
}


Linear::~Linear() {
    delete weight_;
    delete bia_;
}

#endif