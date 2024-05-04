#include<iostream>
#include<torch/torch.h>

using namespace std;

/*
torch::nn::Module : which is a base class for all neural network modules in PyTorch C++.

We will override its methods to build our own NN


*/

struct NeuralNet : torch::nn::Module
{
    // constructor
    NeuralNet(int ip_dim,int hidden_dim,int op_dim)
    {
        input_layer = register_module("FC1",torch::nn::Linear(ip_dim,hidden_dim));
        hidden_layer = register_module("FC2",torch::nn::Linear(hidden_dim,2*hidden_dim));
        output_layer = register_module("FC3",torch::nn::Linear(2*hidden_dim,op_dim));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(input_layer->forward(x));
        x = torch::relu(hidden_layer->forward(x));
        x = torch::softmax(output_layer->forward(x),1);
        return x;
    }

    torch::nn::Linear input_layer{nullptr}; 
    torch::nn::Linear hidden_layer{nullptr}; 
    torch::nn::Linear output_layer{nullptr};

};

int main()
{

    int ipdim = 10;
    int hiddim = 20;
    int opdim = 5;

    auto net = make_shared<NeuralNet>(ipdim,hiddim,opdim);

    torch::Tensor data = torch::randn({8,10});

    // forward pass
    auto output = net->forward(data);

    cout<<"\n\n Forward pass Results = "<<output<<endl;

}