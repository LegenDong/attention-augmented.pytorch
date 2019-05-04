# attention-augmented.pytorch
An unofficial Pytorch Implementation for [Attention Augmented Convolutional Networks
](<https://arxiv.org/abs/1904.09925v1>)

The network structure is as below:

![network structure](./imgs/fig1.png)

The result in paper is as below:

![result](./imgs/fig2.png)

## TODO

Paper said "a minimum of 20 dimensions per head for the keys", the dimensions of keys for heads depends on dk

Before add this, the sum of parameters is 35.8M, and now the sum of parameters is 36.3M, seems right, not sure 


## Reference
[Attention Augmented Convolutional Networks](https://arxiv.org/abs/1904.09925v1)

[wide-resnet.pytorch](https://github.com/meliketoy/wide-resnet.pytorch)