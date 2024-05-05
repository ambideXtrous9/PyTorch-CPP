#include <torch/script.h>
#include<iostream>
#include<torch/torch.h>
#include<stdio.h>
#include <algorithm>
#include <opencv2/opencv.hpp>


using namespace std;

int main()
{

    string path = "/home/ss/STUDY/PyTorch-CPP/MobileNetV2/Inference/TestImages/cat3.jpg";

    vector<string> classes = {"Cat","Dog"};

    cv::Mat img = cv::imread(path);

    cv::resize(img, img, cv::Size(28, 28));

    cv::namedWindow("Display Image", cv::WindowFlags::WINDOW_AUTOSIZE); 
	cv::imshow("Display Image", img); 
	cv::waitKey(0); 

    torch::Tensor imageTensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
    imageTensor = imageTensor.permute({2, 0, 1}).to(torch::kFloat32).div_(255).unsqueeze(0);

    auto net = torch::jit::load("/home/ss/STUDY/PyTorch-CPP/MobileNetV2/DogCatMobileNetv2.pt");

    net.eval();

    vector<torch::jit::IValue> input;
    input.push_back(imageTensor);

    auto out = net.forward(input).toTensor();

    auto max_result = torch::max(out,1);
    torch::Tensor max_values = std::get<0>(max_result);
    float maxProb = max_values.item<float>();

    auto prediction = out.argmax(1);
    int predictedClassIndex = prediction.item<int>();


    cout<<"\n\nPredicted Class = "<<classes[predictedClassIndex]<<" with Prob = "<<maxProb<<endl;

}

