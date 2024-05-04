#include<iostream>
#include<torch/torch.h>
#include<opencv2/opencv.hpp>
#include<stdio.h>
/*

============================ Must be added to CMakeLists.txt =================================

option(DOWNLOAD_MNIST "Download the MNIST dataset from the internet" ON)
if (DOWNLOAD_MNIST)
  message(STATUS "Downloading MNIST dataset")
  execute_process(
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/../tools/download_mnist.py  // download download_mnist.py and give proper path
      -d ${CMAKE_BINARY_DIR}/data
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()

*/


using namespace std;

cv::Mat TensorToCVMat(torch::Tensor tensor)
{
    // torch.squeeze(input, dim=None, *, out=None) → Tensor
    // Returns a tensor with all the dimensions of input of size 1 removed.
    // tensor.detach
    // Returns a new Tensor, detached from the current graph.
    // permute dimension, 3x700x700 => 700x700x3
    tensor = tensor.detach().permute({1, 2, 0});
    // float to 255 range
    // tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
    // GPU to CPU?, may not needed
    tensor = tensor.to(torch::kCPU);
    // shape of tensor
    int64_t height = tensor.size(0);
    int64_t width = tensor.size(1);

    // Mat takes data form like {0,0,255,0,0,255,...} ({B,G,R,B,G,R,...})
    // so we must reshape tensor, otherwise we get a 3x3 grid
    // tensor = tensor.reshape({width * height * 1});
    // CV_8UC3 is an 8-bit unsigned integer matrix/image with 3 channels
    cv::Mat img = cv::Mat(cv::Size(width, height), CV_8UC3, tensor.data_ptr());

    cv::resize(img, img, cv::Size(300, 300));

    return img.clone();
}

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
          std::move(trainset), torch::data::DataLoaderOptions().drop_last(true).batch_size(8));

    for (auto &batch : *trainloader)
    {
        cout << batch.target << endl; 
        auto record = batch.data[0].clone(); //.squeeze(0);

        cout<<"\n\n Record Shape = "<<record.sizes();
        cout<<"\n\n Data Shape = "<<batch.data.sizes();
        cout<<"\n\n Target Shape = "<<batch.target.sizes();

        cv::Mat image = TensorToCVMat(record);

        cv::namedWindow("Display Image", 1);  
	    cv::imshow("Display Image", image); 
	    cv::waitKey(0);
        
        cout << "\nMax value: " << max(record) << endl;
        cout << "\nMin value: " << min(record) << endl;
        break;
    }
}