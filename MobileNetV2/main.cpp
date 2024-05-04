#include <torch/script.h>
#include "./inc/Dataset.h"


using namespace std;



void Test(auto &testloader,int e,auto &criterion,auto &net)
{
    net.eval();

    double epoch_loss = 0.0;
    int num_correct = 0;

    int data_len = 0;

    for (auto &batch : *testloader)
    {

        // Need to convert parameters to torch::jit from Tensor

        vector<torch::jit::IValue> input;
        input.push_back(batch.data);

        auto out = net.forward(input).toTensor(); // convert back to Tensor
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

        // Need to convert parameters to torch::jit from Tensor

        vector<torch::jit::IValue> input;
        input.push_back(batch.data);

        auto out = net.forward(input).toTensor(); // convert back to Tensor
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


    // Paths to train and test folders
    std::string trainFolderPath = "/home/ss/STUDY/PyTorch-CPP/DogCat_Classifier/PetImages/Train";
    std::string testFolderPath = "/home/ss/STUDY/PyTorch-CPP/DogCat_Classifier/PetImages/Test";

    // Create datasets
    auto trainDataset = CustomDataset(trainFolderPath)
    .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
    .map(torch::data::transforms::Stack<>());


    auto testDataset = CustomDataset(testFolderPath)
    .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
    .map(torch::data::transforms::Stack<>());

    // Get the length of the datasets
    size_t trainDatasetSize = *trainDataset.size();
    size_t testDatasetSize = *testDataset.size();

    std::cout << "\n\nSize of train dataset: " << trainDatasetSize << std::endl;
    std::cout << "Size of test dataset: " << testDatasetSize << std::endl;


    int batch_size = 8;
    
    int num_classes = 2;

    int epoch = 2; 

    // Create data loaders
    // Always use RandomSampler instead of SequentialSampler otherwise Acuuracy will stuck at 50%
    auto trainloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(trainDataset), torch::data::DataLoaderOptions().drop_last(true).batch_size(batch_size));

    // Create data loaders
    auto testloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(testDataset), torch::data::DataLoaderOptions().drop_last(true).batch_size(batch_size));


    std::cout << "Train and Test DataLoader created successfully.\n";



    auto net = torch::jit::load("../MobileNet.pt");

    //=========================== Optimizer and Loss Function =============================//


    auto criterion = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().reduction(torch::kMean));


    // Need to convert parameters from torch::jit to Tensor


    std::vector<at::Tensor> parameters;
    for (const auto &params : net.parameters())
    {
        parameters.push_back(params);
    }

        auto optimizer = torch::optim::AdamW(parameters,torch::optim::AdamWOptions().lr(0.001));


    //=========================== Training =============================//

    int e = 1;

    while(e<=epoch)
    {
        Train(trainloader,e,criterion,optimizer,net);

        Test(testloader,e,criterion,net);

        e += 1;
    }

    std::string save_path = "../trained_mobile_net.pt"; // Specify the path to save the trained model

    try
    {
        // Serialize and save the trained model to the specified file
        net.save(save_path);
        std::cout << "Trained model saved successfully to: " << save_path << std::endl;
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Error saving the model: " << e.what() << std::endl;
    }

    return 0;
}

