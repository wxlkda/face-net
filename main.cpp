#include "facenet.hpp"
#include "dataset.hpp"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

void PrintUsage() {
    std::cout << "Usage: FaceNetCpp [train|evaluate] [model_path] [dataset_path]\n";
}

std::string GetCurrentTime() {
    std::time_t now = std::time(nullptr);
    char buf[80];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return std::string(buf);
}

void LogTraining(const std::string& message) {
    std::ofstream log_file("training_log.txt", std::ios_base::app);
    log_file << GetCurrentTime() << " - " << message << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        PrintUsage();
        return 1;
    }

    std::string mode = argv[1];
    std::string model_path = argv[2];
    std::string dataset_path = argv[3];

    FaceNet facenet;
    facenet.LoadModel(model_path);

    if (mode == "train") {
        if (!fs::exists(dataset_path)) {
            std::cerr << "Dataset path does not exist: " << dataset_path << std::endl;
            return 1;
        }

        LogTraining("Starting training...");
        facenet.Train(dataset_path, 50, 0.001, 0.5); // Added epochs, learning rate, margin
        LogTraining("Training completed.");
    }
    else if (mode == "evaluate") {
        if (!fs::exists(dataset_path)) {
            std::cerr << "Dataset path does not exist: " << dataset_path << std::endl;
            return 1;
        }

        facenet.Evaluate(dataset_path);
    }
    else {
        PrintUsage();
        return 1;
    }

    return 0;
}
