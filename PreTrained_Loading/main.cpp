#include <iostream>
#include <torch/torch.h>

using namespace std;


int main() {
  
  torch::Tensor x = torch::randn({2,3});
  
  cout << x <<endl;
  
  cout<<"\n\n TORCH IS WORKING FINE...!!\n\n";

}
