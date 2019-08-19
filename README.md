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

## Ideas
Fit a one-class SVM. Example...
https://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html#sphx-glr-auto-examples-svm-plot-oneclass-py

Use the "local outlier factor" in scikit-learn"
https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html#sphx-glr-auto-examples-neighbors-plot-lof-outlier-detection-py
