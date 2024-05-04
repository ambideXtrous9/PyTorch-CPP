#pragma once

#include<iostream>
#include<torch/torch.h>
#include<stdio.h>

class MnistConvNet : public torch::nn::Module
{
public:
    MnistConvNet(int num_classes);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential Conv{nullptr};
    torch::nn::Sequential FC{nullptr};
};