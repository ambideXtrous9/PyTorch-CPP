#pragma once

#include <torch/torch.h>
#include <iostream>
#include <filesystem>
#include <random>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

class CustomDataset : public torch::data::Dataset<CustomDataset>
{
private:
    std::vector<std::string> filepaths_;
    std::vector<int64_t> labels_;

public:
    explicit CustomDataset(const std::string &folderPath);
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;
};
