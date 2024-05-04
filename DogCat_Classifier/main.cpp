#include "./inc/Dataset.h"
#include "./inc/prepDataset.h"
#include "./inc/model.h"

using namespace std;


void Test(auto &testloader,int e,auto &criterion,auto &net)
{
    net.eval();

    double epoch_loss = 0.0;
    int num_correct = 0;

    int data_len = 0;

    for (auto &batch : *testloader)
    {
        auto out = net.forward(batch.data);
        auto loss = criterion(out,batch.target);

        epoch_loss += loss.template item<double>() * batch.data.size(0);

        data_len += batch.data.size(0);

        auto prediction = out.argmax(1);
        num_correct += prediction.eq(batch.target).sum().template item<int>();
    }

    auto sample_mean_loss = epoch_loss / data_len;
    auto accuracy = static_cast<double>(num_correct) / data_len;

    cout<<"\n|| Epoch = "<<e<<" : Test Loss = "<<sample_mean_loss<<" || Accuracy = "<<accuracy<<" ||\n";

}

void Train(auto &trainloader,int e,auto &criterion,auto &optimizer,auto &net)
{
    net.train();
    
    double epoch_loss = 0.0;
    int data_len = 0;

    for (auto &batch : *trainloader)
    {
        optimizer.zero_grad();

        auto out = net.forward(batch.data);
        auto loss = criterion(out,batch.target);

        epoch_loss += loss.template item<double>() * batch.data.size(0);

        data_len += batch.data.size(0);

        loss.backward();
        optimizer.step();
    }

    auto sample_mean_loss = epoch_loss / data_len;

    cout<<"\n|| Epoch = "<<e<<" : Train Loss = "<<sample_mean_loss<<" || \n";
}

void CUDA_check()
{
    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "\n\n================ CUDA available! Training on GPU ================\n\n" << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "\n\n================ Training on CPU ================\n\n" << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);
}


int main()
{
    CUDA_check();

    std::string parentFolder = "../PetImages";
    std::string categoryFolder = "Cat";
    std::string trainFolder = "Train";
    std::string testFolder = "Test";

    double trainRatio = 0.8;

    createTrainTestFolders(parentFolder, categoryFolder, trainFolder, testFolder, trainRatio);
    
    categoryFolder = "Dog"; // Repeat the process for the 'dog' category folder
    
    createTrainTestFolders(parentFolder, categoryFolder, trainFolder, testFolder, trainRatio);

    std::cout << "Train and Test folders created successfully.\n";

    // Paths to train and test folders
    std::string trainFolderPath = "../PetImages/Train";
    std::string testFolderPath = "../PetImages/Test";

    // Create datasets
    auto trainDataset = CustomDataset(trainFolderPath).map(torch::data::transforms::Stack<>());
    auto testDataset = CustomDataset(testFolderPath).map(torch::data::transforms::Stack<>());

    // Get the length of the datasets
    size_t trainDatasetSize = *trainDataset.size();
    size_t testDatasetSize = *testDataset.size();

    std::cout << "\n\nSize of train dataset: " << trainDatasetSize << std::endl;
    std::cout << "Size of test dataset: " << testDatasetSize << std::endl;


    int batch_size = 8;
    
    int num_classes = 2;

    int epoch = 20; 

    // Create data loaders
    auto trainloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(trainDataset), torch::data::DataLoaderOptions().drop_last(true).batch_size(batch_size));

    // Create data loaders
    auto testloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(testDataset), torch::data::DataLoaderOptions().drop_last(true).batch_size(batch_size));


    std::cout << "Train and Test DataLoader created successfully.\n";



    auto net = CatDogConvNet(num_classes);

    //=========================== Optimizer and Loss Function =============================//


    auto criterion = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().reduction(torch::kMean));
    
    auto optimizer = torch::optim::AdamW(net.parameters(),torch::optim::AdamWOptions().lr(0.001));

    //=========================== Training =============================//

    int e = 1;

    while(e<=epoch)
    {
        Train(trainloader,e,criterion,optimizer,net);

        Test(testloader,e,criterion,net);

        e += 1;
    }    

    return 0;
}
