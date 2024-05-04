#include "../inc/MobileNetV2.h"

using namespace std;


MobileNetV2ConvNet::MobileNetV2ConvNet(int num_classes, string model_path)
{
        model = torch::jit::load(model_path);

        // Freeze all parameters in the model
        for (auto& param : model.named_parameters()) {
            param.value().set_requires_grad(false);
        }
}

torch::Tensor MobileNetV2ConvNet::forward(torch::Tensor x)
{
    x = Conv->forward(x);
    x = x.view({x.size(0),-1});
    x = FC->forward(x);
    return x;
}