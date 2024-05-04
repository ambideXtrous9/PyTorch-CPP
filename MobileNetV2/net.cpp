#include "iostream"
#include "torch/script.h"

using namespace std;

int main()
{

  torch::jit::script::Module net = torch::jit::load("../MobileNet.pt");

  torch::Tensor x = torch::randn({1,3,28,28});

  vector<torch::jit::IValue> input;
  
  input.push_back(x);
  
  auto out = net.forward(input).toTensor();
  
  cout << out << endl;
  cout << typeid(out).name() <<endl ;
}
