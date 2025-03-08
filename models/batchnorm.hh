#ifndef BATCHNORM_HH
#define BATCHNORM_HH

#include <iostream>

#include "my_matrix.hh"

// 修改思路：将gamma和beta的数据结构全部改为vector
//          从模型中得到running_mean和running_var
//          再稍微修改一下计算方法，就大功告成力！
class BatchNorm {
private:
    std::vector<float> gamma_;
    std::vector<float> beta_;
    std::vector<float> running_mean_;
    std::vector<float> running_var_;

public:
    BatchNorm(
        int input_dim, 
        const std::vector<float> &gdata, const std::vector<float> &bdata,
        const std::vector<float> &rm, const std::vector<float> &rv
    );
    ~BatchNorm() {};
    void forward(const MyMatrix& input, MyMatrix& output);
};


BatchNorm::BatchNorm(
    int input_dim, 
    const std::vector<float> &gdata, const std::vector<float> &bdata,
    const std::vector<float> &rm, const std::vector<float> &rv
) {
    gamma_ = gdata;
    beta_ = bdata;
    running_mean_ = rm;
    running_var_ = rv;
}


void BatchNorm::forward(const MyMatrix& input, MyMatrix& output) {
    if (input.col_width_ != gamma_.size()) {
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
    for (int i = 0; i < input.col_width_; ++i) {
        float rm = running_mean_[i], rv = running_var_[i];
        float gm = gamma_[i], bt = beta_[i];
        rv = std::sqrt(rv + 0.00001);
        float tmp;
        for (int j = 0; j < input.row_width_; ++j) {
            tmp = input.mat_[i][j];
            tmp = (tmp - rm) / rv;
            tmp = tmp*gm + bt;
            output.mat_[i][j] = tmp;
        }
    }
}

#endif