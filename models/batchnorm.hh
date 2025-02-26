#ifndef BATCHNORM_HH
#define BATCHNORM_HH

#include <iostream>

#include "my_matrix.hh"

class BatchNorm {
private:
    MyMatrix* gamma_;
    MyMatrix* beta_;
    float running_mean_, running_var_;
    int cnt_;

public:
    BatchNorm(int input_dim, const std::vector<float> &gdata, const std::vector<float> &bdata);
    ~BatchNorm();
    inline void forward(const MyMatrix& input, MyMatrix& output);
};

BatchNorm::BatchNorm(int input_dim, const std::vector<float> &gdata, const std::vector<float> &bdata) {
    gamma_ = new MyMatrix(input_dim, 1);
    beta_ = new MyMatrix(input_dim, 1);
    gamma_->copy(gdata);
    beta_->copy(bdata);
    cnt_ = 0;
}

BatchNorm::~BatchNorm() {
    delete gamma_;
    delete beta_;
}

#endif