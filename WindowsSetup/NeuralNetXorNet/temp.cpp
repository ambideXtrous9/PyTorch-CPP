#include<iostream>
#include<torch/torch.h>

using namespace std;

int main()
{
    int x = 10;
    int y = 5;
    cout<<"\n\nPredicted = "<<x<<" with prob = "<<y<<endl;

    torch::Tensor data = torch::randn({8,10});

    cout<<"\n\Tensor  = "<<data<<endl;

    return 0;

}
