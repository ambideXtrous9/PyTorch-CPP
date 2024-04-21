#include<iostream>
#include<torch/torch.h>
#include<stdio.h>

using namespace std;

/*
self.conv = nn.Sequential( #1x28x28
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=5), # 8x24x24
            nn.Dropout2d(p=0.4,inplace=True),
            nn.LeakyReLU(negative_slope=0.02,inplace=True),
            nn.BatchNorm2d(num_features=8),
            nn.MaxPool2d(2), # 8x12x12
            nn.Conv2d(in_channels=8,out_channels=32,kernel_size=3), # 32x10x10
            nn.Dropout2d(p=0.4,inplace=True),
            nn.LeakyReLU(negative_slope=0.02,inplace=True),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(2), #32x5x5
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=2), # 64x4x4
            nn.Dropout2d(p=0.4,inplace=True),
            nn.BatchNorm2d(64)
            )

        self.linear = nn.Sequential(
            nn.Linear(in_features=64*4*4,out_features=512),
            nn.LeakyReLU(negative_slope=0.02,inplace=True),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(p=0.4,inplace=True),
            nn.Linear(in_features=512,out_features=config['num_classes']),
            nn.Softmax(dim=1))

*/


struct MnistConvNet : torch::nn::Module
{
    // constructor
    MnistConvNet(int num_classes)
    {
        Conv = register_module("conv_sequential", torch::nn::Sequential({
            {"conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1,8,5))},
            {"dout1", torch::nn::Dropout2d(torch::nn::Dropout2dOptions().p(0.4).inplace(true))},
            {"Lrel1", torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.42).inplace(true))},
            {"BN1", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(8).momentum(0.1))},
            {"MxPl1", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}))},
            {"conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(8,32,3))},
            {"dout2", torch::nn::Dropout2d(torch::nn::Dropout2dOptions().p(0.4).inplace(true))},
            {"Lrel2", torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.42).inplace(true))},
            {"BN2", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32).momentum(0.1))},
            {"MxPl2", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}))},
            {"conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(32,64,2))},
            {"dout3", torch::nn::Dropout2d(torch::nn::Dropout2dOptions().p(0.4).inplace(true))},
            {"Lrel3", torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.42).inplace(true))},
            {"BN3", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64).momentum(0.1))},
        })); 

        FC = register_module("fc_sequential", torch::nn::Sequential({
            {"fc1", torch::nn::Linear(torch::nn::LinearOptions(64*4*4,512))},
            {"fc_Lrel1", torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.42).inplace(true))},
            {"fc_BN1", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512).momentum(0.1))},
            {"fc_dout2", torch::nn::Dropout(torch::nn::DropoutOptions().p(0.4).inplace(true))},
            {"fc2", torch::nn::Linear(torch::nn::LinearOptions(512,num_classes))},
            {"op", torch::nn::Softmax(torch::nn::SoftmaxOptions(1))}
        }));    
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = Conv->forward(x);
        x = x.view({x.size(0),-1});
        x = FC->forward(x);
        return x;
    }

    torch::nn::Sequential Conv{nullptr}; 
    torch::nn::Sequential FC{nullptr}; 

};

void Test(auto &testloader,int e,auto &criterion,auto &net)
{
    net->eval();

    double epoch_loss = 0.0;
    int num_correct = 0;

    int data_len = 0;

    for (auto &batch : *testloader)
    {
        auto out = net->forward(batch.data);
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
    net->train();
    
    double epoch_loss = 0.0;
    int data_len = 0;

    for (auto &batch : *trainloader)
    {
        optimizer.zero_grad();

        auto out = net->forward(batch.data);
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

    int batch_size = 8;
    
    int num_classes = 10;

    int epoch = 20; 

    //=========================== DataLoader =============================//

    std::string data_root{ "data" };

    auto trainset = torch::data::datasets::MNIST(data_root,torch::data::datasets::MNIST::Mode::kTrain)
    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
    .map(torch::data::transforms::Stack<>());
    
    auto trainloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(trainset),torch::data::DataLoaderOptions().drop_last(true).batch_size(8));


    auto testset = torch::data::datasets::MNIST(data_root,torch::data::datasets::MNIST::Mode::kTest)
    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
    .map(torch::data::transforms::Stack<>());
    
    auto testloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(testset),torch::data::DataLoaderOptions().drop_last(true).batch_size(8));


    auto net = make_shared<MnistConvNet>(num_classes);

    //=========================== Optimizer and Loss Function =============================//


    auto criterion = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().reduction(torch::kMean));
    
    auto optimizer = torch::optim::AdamW(net->parameters(),torch::optim::AdamWOptions().lr(0.001));

    int e = 1;

    while(e<=epoch)
    {
        Train(trainloader,e,criterion,optimizer,net);

        Test(testloader,e,criterion,net);

        e += 1;
    }

    return 0;
    
}