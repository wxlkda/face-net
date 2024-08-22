// dataset.cpp

#include "dataset.hpp"
#include <dirent.h>
#include <iostream>

std::vector<std::pair<std::string, int>> Dataset::LoadDataset(const std::string& dataset_path) {
    std::vector<std::pair<std::string, int>> dataset;
    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(dataset_path.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_type == DT_DIR) {
                std::string person_name = ent->d_name;
                if (person_name == "." || person_name == "..") continue;

                std::string person_path = dataset_path + "/" + person_name;
                DIR* person_dir;
                struct dirent* person_ent;
                if ((person_dir = opendir(person_path.c_str())) != NULL) {
                    int label = std::hash<std::string>{}(person_name);
                    while ((person_ent = readdir(person_dir)) != NULL) {
                        if (person_ent->d_type == DT_REG) {
                            std::string file_name = person_ent->d_name;
                            dataset.push_back({person_path + "/" + file_name, label});
                        }
                    }
                    closedir(person_dir);
                }
            }
        }
        closedir(dir);
    }
    return dataset;
}

