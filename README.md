# IPSW-RFI
IPSW 2019 work

So far, this is just preliminary presentation stuff.

## Data structure
To check out the data structure,

`jupyter notebook` 

and select the "Exploring [...].ipynb" notebook. Then you'll have to change the `filename=....h5` line.

## Plotting bounding boxes

`plot.py ../path/to/h5File.h5 --draw_boxes`

## Literature
[O'Shea convolutional modulation recognition](https://arxiv.org/pdf/1602.04105.pdf)

[O'Shea autoencoder signal identification](https://arxiv.org/pdf/1611.00303.pdf)

[Unsupervised structured signal identification](https://arxiv.org/pdf/1604.07078.pdf)

[Modulation identification using higher order cumulants](./docs/1-s2.0-S1874490716301094-main.pdf)

[Local subspace clustering](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.5825&rep=rep1&type=pdf)

[Clustering on autoencoder lower dimensional representation](https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf)

## Ideas
Perhaps an autoencoder could work. Would need to train on a bunch of raw data. After training, we would pass new data x throught he autoencoder model and obtain f(x). If the mean-squared error distance between x and f(x) is high, then x is probably anomalous. We could then add x to the training data so that the model f could learn from the anomalies. Problem is the input data x must always be the same shape. The pro is that this could be implemented using a GPU via the PyTorch package.
Paper: https://arxiv.org/pdf/1807.08316.pdf

Some good references on coding autoencoders using the PyTorch package:
https://medium.com/@vaibhaw.vipul/building-autoencoder-in-pytorch-34052d1d280c
https://github.com/L1aoXingyu/pytorch-beginner/tree/master/08-AutoEncoder

PyTorch is a great Machine Learning python library. It allows for easy backpropagation which is needed for training. Futhermore, it enables GPU usage for faster computations. Documentation: https://pytorch.org
