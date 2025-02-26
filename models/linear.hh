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
        const std::vector<std::vector<float>> &wdata, const std::vector<float> &bdata
    );
    ~Linear();
    inline void forward(const MyMatrix& input, MyMatrix& output);
};

Linear::Linear(
    int input_dim, int output_dim, 
    const std::vector<std::vector<float>> &wdata, const std::vector<float> &bdata
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

Linear::~Linear() {
    delete weight_;
    delete bia_;
}

#endif