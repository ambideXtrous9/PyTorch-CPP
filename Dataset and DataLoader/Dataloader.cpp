#include<iostream>
#include<torch/torch.h>


using namespace std;

int main()
{
    int batch_size = 8;

    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    std::string data_root{ "data" };

    auto trainset = torch::data::datasets::MNIST(data_root,torch::data::datasets::MNIST::Mode::kTrain)
    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
    .map(torch::data::transforms::Stack<>());
    
    auto trainloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(trainset), batch_size);

    for (auto &batch : *trainloader)
    {
        cout << batch.target << endl; 
        auto data = batch.data.view({batch_size, -1});
        auto record = data[0].clone();

        cout<<"\n\n Data Shape = "<<batch.data.sizes();
        cout<<"\n\n Target Shape = "<<batch.target.sizes();
        
        cout << "\nMax value: " << max(record) << endl;
        cout << "\nMin value: " << min(record) << endl;
        break;
    }
}