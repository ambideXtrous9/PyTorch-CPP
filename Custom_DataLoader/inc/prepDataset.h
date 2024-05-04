#pragma once

#include <string>
#include <torch/torch.h>
#include <iostream>
#include <filesystem>
#include <random>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

void createTrainTestFolders(const std::string &parentFolder, const std::string &categoryFolder, const std::string &trainFolder, const std::string &testFolder, double trainRatio);
