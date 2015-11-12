#!/usr/bin/python

import os

import sys

import numpy as np

import pandas as pd

from scipy import optimize

import random

import matplotlib.pyplot as plt

from mat4py import loadmat

from scipy import optimize


def randset(start, stop, count, seed_val=0):
    random.seed(seed_val)

    ret = []

    for indx in range(count):
        ret.append(random.randint(start, stop))

    return ret


class mlex3(object):
    def __init__(self, infile):
        self.img_w = None

        self.img_h = None

        self.infile = infile

        self.X = None

        self.y = None

        self.num_labels = 10

        self.lmd = 1.0

        # Each row is theta for one symbol
        self.all_theta = None


    def sigmoid(self, z):
        '''
        compute and return a matrix with sigmoid - retain
        shape
        Note: this method is copied from earlier objects.
        Decided not to inherit entire object for just this
        method
        '''
        g = np.zeros(z.shape)

        g = np.exp(-z)

        g = g + 1.0

        g = 1.0 / g

        return g


    def displayData(self):
        plt.set_cmap('gray')

        m = self.X.shape[0]

        randlist = randset(0, m-1, 100)

        mini = self.X[randlist, :]

        m = mini.shape[0]

        rows = int(np.floor(np.floor(np.sqrt(m))))

        print 'display rows: %d' % rows

        cols = int(np.ceil(m/rows))

        print 'display cols: %d' % cols

        # padding between immages
        pad = 1

        disp_img = np.ones(((pad+rows*(self.img_h+pad)), (pad+cols*(self.img_w+pad))))

        # Each line of mini is copied to a patch in disp_img
        curr_ex = 0

        for j in range(rows):
            for i in range(cols):
                if curr_ex >= m:
                    break

                curr_img = mini[curr_ex, :]

                maxval = max(abs(curr_img))

                strt_h = pad + j*(self.img_h+pad)

                strt_w = pad + i*(self.img_w+pad)

                disp_img[strt_h:(strt_h+self.img_h), strt_w:(strt_w+self.img_w)] = np.reshape(curr_img, (self.img_h, self.img_w)) / maxval

                curr_ex += 1

            if curr_ex >= m:
                break

        plt.imshow(disp_img.T, extent=[-1,1,-1,1])

        plt.axis('off')


    def setupData(self):
        '''
        Load .mat file, convert X and Y lists to numpy arrays
        '''
        data = {}

        data = loadmat(self.infile)

        self.X = np.array(data['X'])

        self.y = np.array(data['y'])

        n = self.X.shape[1]

        # Making an assumption about square images
        self.img_w = np.sqrt(n)

        self.img_h = self.img_w


    def lrGradFunction(self, theta, X, y, lmd):
        '''
        Label by label cost function and gradient.  The
        y vector is a boolean (equiv to 1s and 0s) when
        output matches label being trained for (one of 10 digits)
        theta is computed one label at a time
        Note: This method is copied from mlcl2_reg.py
        '''
        theta = np.mat(theta)

        X = np.mat(X)

        m = X.shape[0]

        if theta.shape[0] != X.shape[1]:
            theta = theta.T

        p = X * theta

        h_theta = self.sigmoid(p)

        grad = np.zeros(X.shape[1])

        grad = (X.T * (h_theta - y)) / m

        grad[1:] = grad[1:] + (theta.A[1:] * lmd / m)

        return np.ndarray.flatten(grad)


    def lrCostFunction(self, theta, X, y, lmd):
        '''
        Label by label cost function and gradient.  The
        y vector is a boolean (equiv to 1s and 0s) when
        output matches label being trained for (one of 10 digits)
        theta is computed one label at a time
        Note: This method is copied from mlcl2_reg.py
        '''
        theta = np.mat(theta)

        X = np.mat(X)

        m = X.shape[0]

        if theta.shape[0] != X.shape[1]:
            theta = theta.T

        p = X * theta

        h_theta = self.sigmoid(p)

        # Use array form when elem by elem mult is needed
        sq_theta = theta.A * theta.A

        J = (-(y.T * np.log(h_theta)) - ((1.0 - y).T * np.log(1.0 - h_theta))) / m

        J = J + (sq_theta.sum() - sq_theta[0]) * lmd / (m*2)

        return J[0][0]


    def oneVsAll(self, X, y, num_labels, lmd):
        '''
        Trains multiple logistic regression classifiers, on the
        same data.  Training data will have positive data sets
        (matching digit being trained for) and negative data sets
        (other digits).  Each training session uses all data to
        compute one row of theta matrix
        '''
        m = y.shape[0]

        ones = np.ones(m)

        X = np.column_stack((ones, X))

        n = X.shape[1]

        all_theta = np.zeros((num_labels, n))

        for label in xrange(1, num_labels+1):
            match = (y == label)

            init_theta = np.zeros(n)

            # all_theta[label-1,:] = optimize.fmin_cg(self.lrCostFunction, fprime=self.lrGradFunction, x0=init_theta, args=(X, match, lmd), maxiter=200)
            all_theta[label-1,:] = optimize.fmin_cg(self.lrCostFunction, x0=init_theta, args=(X, match, lmd), maxiter=200)

        self.all_theta = all_theta

        return all_theta


    def predictOneVsAll(self, X, num_labels):
        '''
        X has data for all symbols.  Apply num_symbol number of
        theta vectors to X, get num_symbol y predictions.  Choose
        the prediction with highest value (best match).  So the
        num_symbol by m matrix will collapse into a 1 by m matrix
        with the content carrying index of best match
        '''
        m = X.shape[0]

        ones = np.ones(m)

        X = np.column_stack((ones, X))

        pred_all = np.zeros((m, num_labels))

        # Not matrices, so using dot prod
        pred_all = np.dot(X, self.all_theta.T)

        # Should return a m by 1 matrix of symbols
        pred = pred_all.argmax(axis=1)

        pred = pred + 1

        return pred


    def run(self):
        pass



def main():
    # Here, instantiate instance and execute
    ex3 = mlex3('/home/vijaykam/work/mc_learn/matlab/mlclass-ex3/ex3data1.mat')

    ex3.setupData()

    # ex3.displayData()

    # plt.show(block=True)

    # raw_input('Press any key to continue\n')

    ex3.oneVsAll(ex3.X, ex3.y, ex3.num_labels, ex3.lmd)

    pred = ex3.predictOneVsAll(ex3.X, ex3.num_labels)

    print pred


if __name__ == '__main__':
    main()
