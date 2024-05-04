#include "iostream"
#include "torch/script.h"

using namespace std;

int main()
{

  torch::jit::script::Module net = torch::jit::load("../models/net.pt");

  torch::Tensor x = torch::randn({1, 100});

  vector<torch::jit::IValue> input;
  
  input.push_back(x);
  
  auto out = net.forward(input);
  
  cout << out << endl;
  cout << typeid(out).name() <<endl ;
}
