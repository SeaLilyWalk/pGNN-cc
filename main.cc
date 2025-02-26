#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>

#include "models/graphcnn.hh"

void load_model_data(const std::string &path, std::map<std::string, std::vector<std::vector<float>> > &data) {
    std::ifstream get_data(path);
    std::string str;
    int dim, row, col;
    while (std::getline(get_data, str)) {
        std::string name(str);
        data[name] = std::vector<std::vector<float>>();
        std::getline(get_data, str);
        std::stringstream data_size(str);
        data_size >> dim;
        if (dim == 1) {
            data_size >> col;
            row = 1;
        } else {
            data_size >> row >> col;
        }
        for (int i = 0; i < row; ++i) {
            std::getline(get_data, str);
            data[name].push_back(std::vector<float>());
            std::stringstream data_val(str);
            float val;
            for (int j = 0; j < col; ++j) {
                data_val >> val;
                data[name][i].push_back(val);
            }
        }
    }
    get_data.close();
}

int main(int argc, char** argv) {
    // load the model data
    std::string path(argv[1]);
    std::map<std::string, std::vector<std::vector<float>> > data;
    load_model_data(path, data);

    // load the model
    float drop_out = 0.1;
    float learn_esp = 0.3;
    std::string graph_pooling_type = "max";
    std::string neighbor_pooling_type = "max";
    GraphCNN model(data, drop_out, learn_esp, graph_pooling_type, neighbor_pooling_type);

    return 0;
}