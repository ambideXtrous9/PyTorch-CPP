#include<iostream>
#include<torch/torch.h>
#include<torch/script.h>

using namespace std;

int main()
{
    cout<<"\nAutograd : Pytorch gradient calculations\n\n";

    // set the precision
    
    cout<<fixed<<setprecision(8);

    torch::Tensor x = torch::tensor(1.0,torch::requires_grad()); // x
    torch::Tensor w = torch::tensor(2.0,torch::requires_grad()); // w
    torch::Tensor b = torch::tensor(3.0,torch::requires_grad()); // b

    cout<<"\nTensor x = "<<x<<endl;
    cout<<"\nTensor w = "<<w<<endl;
    cout<<"\nTensor b = "<<b<<endl;

    // y = w*x + b

    // build a graph
    auto y = w * x + b;

    cout<<"\n y = w*x + b : "<<y<<endl;

    // calculate gradient  dy/dx = w, dy/dw = x, dy/db = 1
    y.backward();

    // print gradient of every component
    
    cout<<"\nGradient of x = "<<x.grad()<<endl;
    cout<<"\nGradient of w = "<<w.grad()<<endl;
    cout<<"\nGradient of b = "<<b.grad()<<endl;


}