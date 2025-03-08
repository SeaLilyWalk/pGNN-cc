#ifndef MY_MATRIX_HH
#define MY_MATRIX_HH

#include <iostream>
#include <vector>
#include <cmath>

class MyMatrix {
private:
    int row_width_, col_width_;
    float **mat_;
public:
    MyMatrix(int col_wid, int row_wid);
    MyMatrix(const MyMatrix& m);
    ~MyMatrix();
    float get_value(int i, int j);
    int get_row_width();
    int get_col_width();
    float get_min_val(int dim, int idx);
    float get_max_val(int dim, int idx);
    int get_min_idx(int dim, int idx);
    int get_max_idx(int dim, int idx);
    void get_row(int idx, std::vector<float> &row);

    void set_value(float value, int i, int j);
    void copy(const MyMatrix& m);
    void copy(const std::vector<float>& m);
    void check();

    void add(const MyMatrix& a, const MyMatrix &b);
    void sub(const MyMatrix& a, const MyMatrix &b);
    void mult(const MyMatrix& a, const MyMatrix &b);
    void mult(float k);
    void dotMult(const MyMatrix& a, const MyMatrix &b);
    void transpose(const MyMatrix& a);
    void activation(const MyMatrix& input, const std::string& type);

    friend class Linear;
    friend class BatchNorm;
};

MyMatrix::MyMatrix(int col_wid, int row_wid) {
    row_width_ = row_wid;
    col_width_ = col_wid;
    mat_ = new float*[col_wid]();
    for (int i = 0; i < col_wid; ++i)
        mat_[i] = new float[row_wid]();
}

MyMatrix::MyMatrix(const MyMatrix& m) {
    this->row_width_ = m.row_width_;
    this->col_width_ = m.col_width_;
    this->mat_ = new float*[col_width_]();
    for (int i = 0; i < col_width_; ++i) {
        mat_[i] = new float[row_width_]();
        for (int j = 0; j < row_width_; ++j)
            this->mat_[i][j] = m.mat_[i][j];
    }
}

MyMatrix::~MyMatrix() {
    for (int i = 0; i < col_width_; ++i)
        delete [] mat_[i];
    delete [] mat_;
}

inline void MyMatrix::set_value(float value, int i, int j) {
    if (i >= this->col_width_ || j >= this->row_width_) {
        std::cerr << "set value error: out of matrix range!" << std::endl;
        exit(0);
    }
    this->mat_[i][j] = value;
}

inline float MyMatrix::get_value(int i, int j) {
    if (i >= this->col_width_ || j >= this->row_width_) {
        std::cerr << "get value error: out of matrix range!" << std::endl;
        exit(0);
    }
    return this->mat_[i][j];
}

inline int MyMatrix::get_row_width() {
    return this->row_width_;
}

inline int MyMatrix::get_col_width() {
    return this->col_width_;
}

float MyMatrix::get_min_val(int dim, int idx) {
    float re;
    if (dim == 0) {
        if (idx >= row_width_) {
            std::cerr << "get min error: wrong idx!" << std::endl;
            exit(0);
        }
        re = mat_[0][idx];
        for (int i = 1; i < col_width_; ++i)
            re = std::min(re, mat_[i][idx]);
    } else if (dim == 1) {
        if (idx >= col_width_) {
            std::cerr << "get min error: wrong idx!" << std::endl;
            exit(0);
        }
        re = mat_[idx][0];
        for (int i = 1; i < row_width_; ++i)
            re = std::min(re, mat_[idx][i]);
    } else {
        std::cerr << "get min error: dim error!" << std::endl;
        exit(0);
    }
    return re;
}

float MyMatrix::get_max_val(int dim, int idx) {
    float re;
    if (dim == 0) {
        if (idx >= row_width_) {
            std::cerr << "get max error: wrong idx!" << std::endl;
            exit(0);
        }
        re = mat_[0][idx];
        for (int i = 1; i < col_width_; ++i)
            re = std::max(re, mat_[i][idx]);
    } else if (dim == 1) {
        if (idx >= col_width_) {
            std::cerr << "get max error: wrong idx!" << std::endl;
            exit(0);
        }
        re = mat_[idx][0];
        for (int i = 1; i < row_width_; ++i)
            re = std::max(re, mat_[idx][i]);
    } else {
        std::cerr << "get max error: dim error!" << std::endl;
        exit(0);
    }
    return re;
}
    
int MyMatrix::get_min_idx(int dim, int idx) {
    float min_val = 0;
    int re = 0;
    if (dim == 0) {
        if (idx >= row_width_) {
            std::cerr << "get min error: wrong idx!" << std::endl;
            exit(0);
        }
        min_val = mat_[0][idx];
        for (int i = 1; i < col_width_; ++i)
            if (mat_[i][idx] < min_val) {
                min_val = mat_[i][idx];
                re = i;
            }
    } else if (dim == 1) {
        if (idx >= col_width_) {
            std::cerr << "get min error: wrong idx!" << std::endl;
            exit(0);
        }
        min_val = mat_[idx][0];
        for (int i = 1; i < row_width_; ++i)
            if (mat_[idx][i] < min_val) {
                min_val = mat_[idx][i];
                re = i;
            }
    } else {
        std::cerr << "get min error: dim error!" << std::endl;
        exit(0);
    }
    return re;
}

int MyMatrix::get_max_idx(int dim, int idx) {
    float max_val = 0;
    int re = 0;
    if (dim == 0) {
        if (idx >= row_width_) {
            std::cerr << "get max error: wrong idx!" << std::endl;
            exit(0);
        }
        max_val = mat_[0][idx];
        for (int i = 1; i < col_width_; ++i)
            if (mat_[i][idx] > max_val) {
                max_val = mat_[i][idx];
                re = i;
            }
    } else if (dim == 1) {
        if (idx >= col_width_) {
            std::cerr << "get max error: wrong idx!" << std::endl;
            exit(0);
        }
        max_val = mat_[idx][0];
        for (int i = 1; i < row_width_; ++i)
            if (max_val < mat_[idx][i]) {
                max_val = mat_[idx][i];
                re = i;
            }
    } else {
        std::cerr << "get max error: dim error!" << std::endl;
        exit(0);
    }
    return re;
}

void MyMatrix::get_row(int idx, std::vector<float> &row) {
    if (idx >= col_width_) {
        std::cerr << "get row error: wrong idx!" << std::endl;
        exit(0);
    }
    for (int i = 0; i < row_width_; ++i)
        row.push_back(mat_[idx][i]);
}

void MyMatrix::copy(const MyMatrix& m) {
    if (col_width_ != m.col_width_ || row_width_ != m.row_width_) {
        std::cerr << "copy error: copy wrong size!" << std::endl;
        exit(0);
    }
    for (int i = 0; i < col_width_; ++i) 
        for (int j = 0; j < row_width_; ++j)
            this->mat_[i][j] = m.mat_[i][j];
}

void MyMatrix::copy(const std::vector<float>& m) {
    if (col_width_*row_width_ != m.size()) {
        std::cerr << "copy error: copy wrong size!" << std::endl;
        exit(0);
    }
    for (int i = 0; i < col_width_; ++i)
        for (int j = 0; j < row_width_; ++j)
            this->mat_[i][j] = m[i*row_width_ + j];
}

void MyMatrix::add(const MyMatrix& a, const MyMatrix &b) {
    if (this->col_width_ != a.col_width_ || this->col_width_ != b.col_width_) {
        std::cerr << "add error: illegal size of matrix!" << std::endl;
        exit(0);
    }
    if (this->row_width_ != a.row_width_ || this->row_width_ != b.row_width_) {
        std::cerr << "add error: illegal size of matrix!" << std::endl;
        exit(0);
    }
    for (int i = 0; i < col_width_; ++i)
        for (int j = 0; j < row_width_; ++j)
            this->mat_[i][j] = a.mat_[i][j] + b.mat_[i][j];
}

void MyMatrix::sub(const MyMatrix& a, const MyMatrix &b) {
    if (this->col_width_ != a.col_width_ || this->col_width_ != b.col_width_) {
        std::cerr << "sub error: illegal size of matrix!" << std::endl;
        exit(0);
    }
    if (this->row_width_ != a.row_width_ || this->row_width_ != b.row_width_) {
        std::cerr << "sub error: illegal size of matrix!" << std::endl;
        exit(0);
    }
    for (int i = 0; i < col_width_; ++i)
        for (int j = 0; j < row_width_; ++j)
            this->mat_[i][j] = a.mat_[i][j] - b.mat_[i][j];
}

void MyMatrix::mult(const MyMatrix& a, const MyMatrix &b) {
    if (a.row_width_ != b.col_width_) {
        std::cerr << "mult error: illegal size of matrix!" << std::endl;
        std::cout << a.col_width_ << ' ' << a.row_width_ << ' ' << b.col_width_ << ' ' << b.row_width_ << std::endl;
        exit(0);
    }
    if (a.col_width_ != this->col_width_ || b.row_width_ != this->row_width_) {
        std::cerr << "mult error: illegal size of matrix!" << std::endl;
        std::cout << a.col_width_ << ' ' << a.row_width_ << ' ' << b.col_width_ << ' ' << b.row_width_ << std::endl;
        std::cout << this->col_width_ << ' ' << this->row_width_ << std::endl;
        exit(0);
    }
    MyMatrix re(this->col_width_, this->row_width_);
    for (int i = 0; i < this->col_width_; ++i) 
        for (int j = 0; j < this->row_width_; ++j) {
            float m = 0;
            for (int k = 0; k < a.row_width_; ++k)
                m += a.mat_[i][k] * b.mat_[k][j];
            re.mat_[i][j] = m;
        }
    this->copy(re);
}

void MyMatrix::mult(float k) {
    for (int i = 0; i < col_width_; ++i)
        for (int j = 0; j < row_width_; ++j)
            mat_[i][j] *= k;
}

void MyMatrix::dotMult(const MyMatrix& a, const MyMatrix &b) {
    if (this->col_width_ != a.col_width_ || this->col_width_ != b.col_width_) {
        std::cerr << "dot mult error: illegal size of matrix!" << std::endl;
        exit(0);
    }
    if (this->row_width_ != a.row_width_ || this->row_width_ != b.row_width_) {
        std::cerr << "dot mult error: illegal size of matrix!" << std::endl;
        exit(0);
    }
    for (int i = 0; i < col_width_; ++i)
        for (int j = 0; j < row_width_; ++j)
            this->mat_[i][j] = a.mat_[i][j] * b.mat_[i][j];
}

void MyMatrix::transpose(const MyMatrix& a) {
    if (a.col_width_ != this->row_width_ || a.row_width_ != this->col_width_) {
        std::cerr << "transpose error: illegal size of matrix!" << std::endl;
        exit(0);
    }
    for (int i = 0; i < this->col_width_; ++i)
        for (int j = 0; j < this->row_width_; ++j)
            this->mat_[i][j] = a.mat_[j][i];
}

void MyMatrix::activation(const MyMatrix& input, const std::string& type) {
    if (input.col_width_ != this->col_width_ || input.row_width_ != this->row_width_) {
        std::cerr << "activation error: wrong matrix size!" << std::endl;
        exit(0);
    }
    for (int i = 0; i < col_width_; ++i) 
        for (int j = 0; j < row_width_; ++j) {
            float m = input.mat_[i][j];
            if (type == "sigmoid") {
                if (m > 10)
                    m = 10;
                float tmp = exp(m);
                m = tmp / (1 + tmp);
            }
            else if (type == "tanh") {
                if (m > 10)
                    m = 10;
                m = (exp(m) - 1/exp(m)) / (exp(m) + 1/exp(m));
            }
            else if (type == "ReLU")
                m = m < 0 ? 0 : m;
            this->mat_[i][j] = m;
        }
}

void MyMatrix::check() {
    for (int i = 0; i < row_width_; ++i) {
        for (int j = 0; j < col_width_; ++j)
            std::cout << mat_[j][i] << ' ';
        std::cout << std::endl;
    }
}

#endif
