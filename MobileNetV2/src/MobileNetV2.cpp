#include "../inc/MobileNetV2.h"

using namespace std;


MobileNetV2ConvNet::MobileNetV2ConvNet(int num_classes, string model_path)
{
        model = torch::jit::load(model_path);

        // Freeze all parameters in the model
        for (auto& param : model.named_parameters()) {
            param.value().set_requires_grad(false);
        }

        // Get the feature extractor part of the model
        auto FC = model->get_module("features");

        // Define and append custom layers
        register_module("fc_sequential", torch::nn::Sequential({feature_extractor_.register_module("fc1", torch::nn::Linear(1280, 512)),
        feature_extractor_.register_module("relu", torch::nn::Functional(torch::relu));
        feature_extractor_.register_module("fc2", torch::nn::Linear(512, num_classes_));
   
}

torch::Tensor MobileNetV2ConvNet::forward(torch::Tensor x)
{
    x = Conv->forward(x);
    x = x.view({x.size(0),-1});
    x = FC->forward(x);
    return x;
}