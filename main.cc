#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>

#include "models/graphcnn.hh"
#include "models/my_matrix.hh"
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


void test(const std::vector<S2VGraph*>& data, int batch_size) {
    int data_l = data.size();
    for (int begin_idx = 0; begin_idx < data_l; begin_idx += batch_size) {
        std::vector<S2VGraph*> batch_data;
        for (int i = 0; i < batch_size && i+begin_idx < data_l; ++i)
            batch_data.push_back(data[i+begin_idx]);
    }
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
    std::string graph_pooling_type = "sum";
    std::string neighbor_pooling_type = "sum";
    GraphCNN model(
        model_data, false, 
        graph_pooling_type, neighbor_pooling_type
    );

    // load train data and test data
    std::string data_path(argv[2]);
    std::vector<S2VGraph*> graph_list;
    int label_sum = 0, tag_sum = 0;
    loadData(data_path, 0, graph_list, label_sum, tag_sum);

    int g_list_size = graph_list.size();
    int output_dim = model.get_output_dim();
    int correct = 0;
    for (int i = 0; i < g_list_size; i += 64) {
        std::vector<S2VGraph*> batch;
        int batch_size = 64;
        if (g_list_size - i < 64)
            batch_size = g_list_size - i;
        for (int j = i; j < i+batch_size; ++j) 
            batch.push_back(graph_list[j]);
        MyMatrix output(output_dim, batch_size);
        model.forward(batch, tag_sum, output);
        int pre;
        for (int j = i; j < i+batch_size; ++j) {
            pre = output.get_max_idx(0, j-i);
            if (pre == graph_list[j]->get_label())
                correct++;
        }
    }
    float accuracy =  correct;
    accuracy /= float(g_list_size);
    std::cout << "accuracy: " << accuracy << std::endl;

    deleteData(graph_list);

    return 0;
}