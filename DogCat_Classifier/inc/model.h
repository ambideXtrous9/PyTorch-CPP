#pragma once

#include<iostream>
#include<torch/torch.h>
#include<stdio.h>
#include<torch/script.h>


class CatDogConvNet : public torch::nn::Module
{
public:
    CatDogConvNet(int num_classes);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential Conv{nullptr};
    torch::nn::Sequential FC{nullptr};
};