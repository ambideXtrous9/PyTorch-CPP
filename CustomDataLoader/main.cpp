#include "./inc/Dataset.h"
#include "./inc/prepDataset.h"

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

    std::cout << "Train and Test folders created successfully.\n";

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
