#ifndef BATCHNORM_HH
#define BATCHNORM_HH

#include <iostream>

#include "my_matrix.hh"

class BatchNorm {
private:
    MyMatrix* gamma_;
    MyMatrix* beta_;

public:
    BatchNorm(
        int input_dim, 
        const std::vector<float> &gdata, 
        const std::vector<float> &bdata
    );
    ~BatchNorm();
    void forward(const MyMatrix& input, MyMatrix& output);
};


BatchNorm::BatchNorm(
    int input_dim, 
    const std::vector<float> &gdata, 
    const std::vector<float> &bdata
) {
    gamma_ = new MyMatrix(input_dim, 1);
    beta_ = new MyMatrix(input_dim, 1);
    gamma_->copy(gdata);
    beta_->copy(bdata);
}


void BatchNorm::forward(const MyMatrix& input, MyMatrix& output) {
    if (input.col_width_ != gamma_->col_width_) {
        std::cerr << "batch norm error: wrong size of input!" << std::endl;
        exit(0);
    }
    if (
        input.col_width_ != output.col_width_ || 
        input.row_width_ != output.row_width_
    ) {
        std::cerr << "batch norm error: wrong size of input!" << std::endl;
        exit(0);
    }
    for (int i = 0; i < input.row_width_; ++i) {
        float mean = 0, mean_2 = 0, tmp = 0;
        for (int j = 0; j < input.col_width_; ++j) {
            tmp = input.mat_[j][i];
            mean += tmp;
            mean_2 += tmp*tmp;
        }
        mean /= float(input.col_width_);
        mean_2 /= float(input.col_width_);
        float var = mean_2 - mean;
        var = std::sqrtf(var);
        var += 0.00001;
        for (int j = 0; j < input.col_width_; ++j) {
            tmp = input.mat_[j][i];
            tmp -= mean;
            tmp /= var;
            tmp *= gamma_->mat_[j][0];
            tmp += beta_->mat_[j][0];
            output.set_value(tmp, j, i);
        }
    }
}


BatchNorm::~BatchNorm() {
    delete gamma_;
    delete beta_;
}

#endif