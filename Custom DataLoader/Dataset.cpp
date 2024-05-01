#include <torch/torch.h>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

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
    // Paths to train and test folders
    std::string trainFolderPath = "/home/ss/STUDY/PyTorch-CPP/Custom DataLoader/PetImages/Train";
    std::string testFolderPath = "/home/ss/STUDY/PyTorch-CPP/Custom DataLoader/PetImages/Test";

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
