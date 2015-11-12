This is a numpy based implementation of logistical regression
from Andrew Ng's Coursera course on machine learning.

The expected input is a series of 20x20 grey scale images, with
centered, hand-written numbers in them.  The X vector is basically
rows of these 20x20 images, with the pixels flattened out on one
400 element row.  The y vector carries the labels for each image
Each label is 1+ the number shown in the image (to handle MATLAB
indexing).

A "one-versus-all" logistic regression is done for each number
symbol.

TODO:
Using my own "lrGradientFunction" in the optimization function did
not work even after trying some online suggestions (like flattening
out the gradient vector.
