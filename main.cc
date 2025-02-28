#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>

#include "models/graphcnn.hh"
#include "s2vgraph.hh"
#include "util.hh"


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


void deleteData(std::vector<S2VGraph*>& data) {
    for (int i = 0; i < data.size(); ++i)
        delete data[i];
}


int main(int argc, char** argv) {
    // load the model data
    std::string model_path(argv[1]);
    std::map<std::string, std::vector<std::vector<float>> > model_data;
    load_model_data(model_path, model_data);

    // load the model
    float drop_out = 0.1;
    float learn_esp = 0.3;
    std::string graph_pooling_type = "max";
    std::string neighbor_pooling_type = "max";
    GraphCNN model(
        model_data, drop_out, learn_esp, 
        graph_pooling_type, neighbor_pooling_type
    );

    // load train data and test data
    std::string data_path(argv[2]);
    std::vector<S2VGraph*> graph_list;
    int label_sum = 0;
    loadData(data_path, 0, graph_list, label_sum);
    std::vector<S2VGraph*> train_list, test_list;
    separateData(graph_list, 9, train_list, test_list);

    // because the space is used by both the graph list and separated list
    // so only delete the graph list 
    deleteData(graph_list);

    return 0;
}