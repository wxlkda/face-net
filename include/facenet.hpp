// facenet.hpp

#ifndef FACENET_HPP
#define FACENET_HPP

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>

class FaceNet {
public:
    FaceNet();
    void LoadModel(const std::string& model_path);
    void Train(const std::string& dataset_path, int epochs, float learning_rate, float margin);
    void Evaluate(const std::string& dataset_path);
    std::vector<float> GetEmbedding(const cv::Mat& image);

private:
    std::unique_ptr<tensorflow::Session> session;
    tensorflow::GraphDef graph_def;

    std::vector<std::tuple<tensorflow::Tensor, tensorflow::Tensor, tensorflow::Tensor>> GenerateTriplets(const std::vector<std::pair<std::string, int>>& images);
    float TripletLoss(const tensorflow::Tensor& anchor, const tensorflow::Tensor& positive, const tensorflow::Tensor& negative, float alpha);
    tensorflow::Tensor PreprocessImage(const cv::Mat& image);
    void LogTraining(const std::string& message);
};

#endif // FACENET_HPP

