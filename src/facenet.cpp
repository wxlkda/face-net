#include "facenet.hpp"
#include "dataset.hpp"
#include <iostream>
#include <cmath>
#include <fstream>
#include <random>
#include <numeric>

FaceNet::FaceNet() {
    tensorflow::SessionOptions options;
    tensorflow::ConfigProto& config = options.config;
    config.set_allow_soft_placement(true);
    config.mutable_gpu_options()->set_allow_growth(true);
    session.reset(tensorflow::NewSession(options));
}

void FaceNet::LoadModel(const std::string& model_path) {
    tensorflow::Status load_graph_status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), model_path, &graph_def);
    if (!load_graph_status.ok()) {
        throw std::runtime_error("Failed to load graph: " + load_graph_status.ToString());
    }
    session->Create(graph_def);
}

tensorflow::Tensor FaceNet::PreprocessImage(const cv::Mat& image) {
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(160, 160));
    resized_image.convertTo(resized_image, CV_32F);
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 160, 160, 3}));
    auto tensor_mapped = input_tensor.tensor<float, 4>();
    for (int y = 0; y < 160; ++y) {
        for (int x = 0; x < 160; ++x) {
            for (int c = 0; c < 3; ++c) {
                tensor_mapped(0, y, x, c) = resized_image.at<cv::Vec3f>(y, x)[c];
            }
        }
    }
    return input_tensor;
}

std::vector<std::tuple<tensorflow::Tensor, tensorflow::Tensor, tensorflow::Tensor>> FaceNet::GenerateTriplets(const std::vector<std::pair<std::string, int>>& images) {
    std::vector<std::tuple<tensorflow::Tensor, tensorflow::Tensor, tensorflow::Tensor>> triplets;
    std::unordered_map<int, std::vector<int>> label_to_indices;
    
    for (size_t i = 0; i < images.size(); ++i) {
        label_to_indices[images[i].second].push_back(i);
    }

    std::random_device rd;
    std::mt19937 g(rd());

    for (auto& [label, indices] : label_to_indices) {
        std::shuffle(indices.begin(), indices.end(), g);
    }

    for (auto& [label, indices] : label_to_indices) {
        for (size_t i = 0; i < indices.size() - 1; ++i) {
            int anchor_index = indices[i];
            int positive_index = indices[i + 1];

            int negative_index;
            do {
                negative_index = g() % images.size();
            } while (images[negative_index].second == label);

            tensorflow::Tensor anchor = PreprocessImage(cv::imread(images[anchor_index].first));
            tensorflow::Tensor positive = PreprocessImage(cv::imread(images[positive_index].first));
            tensorflow::Tensor negative = PreprocessImage(cv::imread(images[negative_index].first));

            triplets.push_back({anchor, positive, negative});
        }
    }

    return triplets;
}

float FaceNet::TripletLoss(const tensorflow::Tensor& anchor, const tensorflow::Tensor& positive, const tensorflow::Tensor& negative, float alpha) {
    auto anchor_flat = anchor.flat<float>();
    auto positive_flat = positive.flat<float>();
    auto negative_flat = negative.flat<float>();

    float positive_distance = 0.0f;
    float negative_distance = 0.0f;

    for (int i = 0; i < anchor_flat.size(); ++i) {
        positive_distance += std::pow(anchor_flat(i) - positive_flat(i), 2);
        negative_distance += std::pow(anchor_flat(i) - negative_flat(i), 2);
    }

    positive_distance = std::sqrt(positive_distance);
    negative_distance = std::sqrt(negative_distance);

    return std::max(0.0f, positive_distance - negative_distance + alpha);
}

void FaceNet::Train(const std::string& dataset_path, int epochs, float learning_rate, float margin) {
    auto dataset = Dataset::LoadDataset(dataset_path);

    tensorflow::Tensor learning_rate_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape());
    learning_rate_tensor.scalar<float>()() = learning_rate;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto triplets = GenerateTriplets(dataset);

        float epoch_loss = 0.0f;
        int batch_count = 0;

        for (const auto& triplet : triplets) {
            tensorflow::Tensor anchor, positive, negative;
            std::tie(anchor, positive, negative) = triplet;
            float loss = TripletLoss(anchor, positive, negative, margin);

            std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict = {
                {"anchor_input:0", anchor},
                {"positive_input:0", positive},
                {"negative_input:0", negative},
                {"learning_rate:0", learning_rate_tensor}
            };

            std::vector<tensorflow::Tensor> outputs;
            tensorflow::Status run_status = session->Run(feed_dict, {}, {"train_op"}, &outputs);
            if (!run_status.ok()) {
                std::cerr << "Training failed: " << run_status.ToString() << std::endl;
                return;
            }

            epoch_loss += loss;
            batch_count++;

            if (batch_count % 10 == 0) {
                std::cout << "Epoch: " << epoch + 1 << ", Batch: " << batch_count << ", Loss: " << loss << std::endl;
            }
        }

        std::cout << "Epoch " << epoch + 1 << " completed with average loss: " << epoch_loss / batch_count << std::endl;
        LogTraining("Epoch " + std::to_string(epoch + 1) + " average loss: " + std::to_string(epoch_loss / batch_count));
    }
}

void FaceNet::Evaluate(const std::string& dataset_path) {
    auto dataset = Dataset::LoadDataset(dataset_path);
    int correct = 0;

    for (size_t i = 0; i < dataset.size(); ++i) {
        tensorflow::Tensor image_tensor = PreprocessImage(cv::imread(dataset[i].first));
        std::vector<tensorflow::Tensor> outputs;
        session->Run({{"input:0", image_tensor}}, {"embeddings:0"}, {}, &outputs);
        auto embedding_flat = outputs[0].flat<float>();

        int predicted_label = -1;
        float min_distance = std::numeric_limits<float>::max();
        for (size_t j = 0; j < dataset.size(); ++j) {
            if (i == j) continue;
            tensorflow::Tensor compare_image_tensor = PreprocessImage(cv::imread(dataset[j].first));
            std::vector<tensorflow::Tensor> compare_outputs;
            session->Run({{"input:0", compare_image_tensor}}, {"embeddings:0"}, {}, &compare_outputs);
            auto compare_embedding_flat = compare_outputs[0].flat<float>();

            float distance = 0.0f;
            for (int k = 0; k < embedding_flat.size(); ++k) {
                distance += std::pow(embedding_flat(k) - compare_embedding_flat(k), 2);
            }
            distance = std::sqrt(distance);

            if (distance < min_distance) {
                min_distance = distance;
                predicted_label = dataset[j].second;
            }
        }

        if (predicted_label == dataset[i].second) {
            correct++;
        }

        if (i % 100 == 0) {
            std::cout << "Evaluated " << i + 1 << "/" << dataset.size() << " images. Current accuracy: " << static_cast<float>(correct) / (i + 1) << std::endl;
        }
    }

    float accuracy = static_cast<float>(correct) / dataset.size();
    std::cout << "Final accuracy: " << accuracy << std::endl;
    LogTraining("Evaluation accuracy: " + std::to_string(accuracy));
}
