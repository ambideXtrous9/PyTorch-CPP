#pragma once

#include<iostream>
#include<torch/torch.h>
#include<stdio.h>
#include<torch/script.h>


class MobileNetV2ConvNet : public torch::nn::Module
{
public:
    MobileNetV2ConvNet(int num_classes,string model_path);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::jit::script::Module model;
    torch::nn::Sequential FC{nullptr};
};