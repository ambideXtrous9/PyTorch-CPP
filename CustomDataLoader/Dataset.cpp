#include <torch/torch.h>
#include <iostream>
#include <filesystem>
#include <random>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

void createTrainTestFolders(const std::string &parentFolder, const std::string &categoryFolder, const std::string &trainFolder, const std::string &testFolder, double trainRatio)
{
    fs::path catFolder = fs::path(parentFolder) / categoryFolder;
    fs::path trainCatFolder = fs::path(parentFolder) / trainFolder / categoryFolder;
    fs::path testCatFolder = fs::path(parentFolder) / testFolder / categoryFolder;

    // Create train and test folders if they don't exist
    fs::create_directories(trainCatFolder);
    fs::create_directories(testCatFolder);

    // Collect all image files
    std::vector<fs::path> imageFiles;
    for (const auto &entry : fs::directory_iterator(catFolder))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg")
        {
            imageFiles.push_back(entry.path());
        }
    }

    // Shuffle the image files
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(imageFiles.begin(), imageFiles.end(), g);

    // Determine number of images for training
    size_t numTrainImages = static_cast<size_t>(trainRatio * imageFiles.size());

    // Copy images to train folder
    for (size_t i = 0; i < numTrainImages; ++i)
    {
        fs::copy_file(imageFiles[i], trainCatFolder / imageFiles[i].filename(), fs::copy_options::overwrite_existing);
    }

    // Copy remaining images to test folder
    for (size_t i = numTrainImages; i < imageFiles.size(); ++i)
    {
        fs::copy_file(imageFiles[i], testCatFolder / imageFiles[i].filename(), fs::copy_options::overwrite_existing);
    }
}

// Custom dataset class for loading images
class CustomDataset : public torch::data::Dataset<CustomDataset>
{
private:
    std::vector<std::string> filepaths_;
    std::vector<int64_t> labels_;

public:
    explicit CustomDataset(const std::string &folderPath)
    {
        for (const auto &entry : fs::directory_iterator(folderPath))
        {
            if (entry.is_directory())
            {
                // If the entry is a directory, iterate through its contents
                for (const auto &fileEntry : fs::directory_iterator(entry.path()))
                {

                    if (fileEntry.is_regular_file() && fileEntry.path().extension() == ".jpg")
                    {

                        filepaths_.push_back(fileEntry.path());
                        // Assuming folder name is the label (0 for cat, 1 for dog)

                        if (entry.path().filename() == "Cat")
                        {
                            labels_.push_back(0);
                        }
                        else
                        {
                            labels_.push_back(1);
                        }
                    }
                }
            }
        }
    }

    torch::data::Example<> get(size_t index) override
    {
        cv::Mat img = cv::imread(filepaths_[index]);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cv::resize(img, img, cv::Size(224, 224));
        torch::Tensor imageTensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
        imageTensor = imageTensor.permute({2, 0, 1}).to(torch::kFloat32).div_(255);
        int64_t label = labels_[index];
        return {imageTensor.clone(), torch::tensor(label)};
    }

    torch::optional<size_t> size() const override
    {
        return filepaths_.size();
    }
};

int main()
{

    std::string parentFolder = "/home/ss/STUDY/PyTorch-CPP/CustomDataLoader/PetImages";
    std::string categoryFolder = "Cat";
    std::string trainFolder = "Train";
    std::string testFolder = "Test";
    double trainRatio = 0.8;

    createTrainTestFolders(parentFolder, categoryFolder, trainFolder, testFolder, trainRatio);

    categoryFolder = "Dog"; // Repeat the process for the 'dog' category folder
    createTrainTestFolders(parentFolder, categoryFolder, trainFolder, testFolder, trainRatio);

    std::cout << "\n\nTrain and Test folders created successfully.\n";

    // Paths to train and test folders
    std::string trainFolderPath = "/home/ss/STUDY/PyTorch-CPP/CustomDataLoader/PetImages/Train";
    std::string testFolderPath = "/home/ss/STUDY/PyTorch-CPP/CustomDataLoader/PetImages/Test";

    // Create datasets
    auto trainDataset = CustomDataset(trainFolderPath).map(torch::data::transforms::Stack<>());
    auto testDataset = CustomDataset(testFolderPath).map(torch::data::transforms::Stack<>());
    ;

    // Get the length of the datasets
    size_t trainDatasetSize = *trainDataset.size();
    size_t testDatasetSize = *testDataset.size();

    std::cout << "\n\nSize of train dataset: " << trainDatasetSize << std::endl;
    std::cout << "Size of test dataset: " << testDatasetSize << std::endl;

    // Create data loaders
    auto trainloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(trainDataset), torch::data::DataLoaderOptions().drop_last(true).batch_size(8));

    // Create data loaders
    auto testloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(testDataset), torch::data::DataLoaderOptions().drop_last(true).batch_size(8));

    for (auto &batch : *testloader)
    {

        auto x = batch.data;
        auto y = batch.target;

        auto record = batch.data[0].clone();

        std::cout << "\n\n Record Shape = " << record.sizes() << std::endl;
        std::cout << "\n\n Data Shape = " << batch.data.sizes() << std::endl;
        std::cout << "\n\n Target Shape = " << batch.target.sizes() << std::endl;

        break;
    }

    return 0;
}
