#include<iostream>

#include<torch/torch.h>

using namespace std;

/*
torch::nn::Module : which is a base class for all neural network modules in PyTorch C++.

We will override its methods to build our own NN


*/

struct XORNN : torch::nn::Module
{
    // constructor
    XORNN(int ip_dim,int hidden_dim,int op_dim)
    {
        input_layer = register_module("FC1",torch::nn::Linear(ip_dim,hidden_dim));
        hidden_layer = register_module("FC2",torch::nn::Linear(hidden_dim,20));
        output_layer = register_module("FC3",torch::nn::Linear(20,op_dim));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(input_layer->forward(x));
        x = torch::relu(hidden_layer->forward(x));
        x = torch::sigmoid(output_layer->forward(x));
        return x;
    }

    torch::nn::Linear input_layer{nullptr}; 
    torch::nn::Linear hidden_layer{nullptr}; 
    torch::nn::Linear output_layer{nullptr};

};

int main()
{

    int ipdim = 2;
    int hiddim = 4;
    int opdim = 1;

    auto xornet = make_shared<XORNN>(ipdim,hiddim,opdim);

 

    torch::Tensor xor_X = torch::tensor({{0, 0}, {0, 1}, {1, 0}, {1, 1}}, torch::kFloat32);

    torch::Tensor xor_Y = torch::tensor({{0}, {1}, {1}, {0}},torch::kFloat32);


    // create the loss function

    torch::nn::BCELoss criterion;

    // create the optimizer

    torch::optim::Adam optimizer(xornet->parameters());

    int epoch = 20;

    int e = 1;


    while(e<=epoch)
    {
        // forward pass
        auto y_hat = xornet->forward(xor_X);

        // calculate loss

        auto loss = criterion(y_hat,xor_Y);

        // backward pass
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        cout<<"\nLoss = "<<loss.item()<<" at epoch : "<<e<<endl;

        e += 1;
    }

    cout<<"\n\n Training of XOR Net Completed..\n\n";

    cout<<"\n\n Predicttion using Trained Model \n\n";

    torch::Tensor X = torch::tensor({{0, 1}}, torch::kFloat32);

    float Th = 0.5;

    torch::Tensor prediction = xornet->forward(X);

    float pd = prediction.item<float>();

    bool predict = (pd > Th);

    cout<<"\n\nPredicted = "<<predict<<" with prob = "<<pd<<endl;

}