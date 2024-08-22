// dataset.hpp

#ifndef DATASET_HPP
#define DATASET_HPP

#include <string>
#include <vector>
#include <utility>

class Dataset {
public:
    static std::vector<std::pair<std::string, int>> LoadDataset(const std::string& dataset_path);
};

#endif // DATASET_HPP

