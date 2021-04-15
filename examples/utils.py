#  Created byMartin.cz

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm


def predict(model, X, Y, threshold=0.5):
    """Predict on test data."""
    
    # get shape
    m = X.shape[1]
    C = Y.shape[0]
    
    # predict by model
    A = model.forward(X)
    
    # convert to 0/1 predictions
    p = A > threshold
    
    # calc accuracy
    accuracy = np.sum((p == Y))/m/C * 100
    
    # print results
    # print("Predictions: %s" % str(p))
    print("Accuracy: %.2f %%" % accuracy)


def plot_costs(costs, rate, epochs):
    """Plots learning curve of the model."""
    
    # init plot
    plt.figure()
    plt.title("Learning rate: " + str(rate))
    plt.ylabel('Cost')
    plt.xlabel('Iterations ')
    
    # plot curve
    plt.plot(np.squeeze(costs))
    
    # set x-ticks
    locs, labels = plt.xticks()
    steps = int(epochs / len(costs))
    plt.xticks(locs[1:-1], tuple(np.array(locs[1:-1], dtype='int')*steps))
    plt.xticks()
    
    # show plot
    plt.show()


def plot_boundaries(model, X, Y, threshold=0.5):
    """Plots decision boundaries."""
    
    NX = 1000
    NY = 1000
    
    # transpose
    X = X.T
    Y = Y.T
    C = Y.shape[1]
    
    # get range
    min_x = min(X[:, 0])
    max_x = max(X[:, 0])
    min_y = min(X[:, 1])
    max_y = max(X[:, 1])
    
    f = 0.3
    x_range = (min_x-f*(max_x-min_x), max_x+f*(max_x-min_x))
    y_range = (min_y-f*(max_y-min_y), max_y+f*(max_y-min_y))
    
    # generate a grid of points
    xs = np.linspace(x_range[0], x_range[1], NX)
    ys = np.linspace(y_range[1], y_range[0], NY)
    xx, yy = np.meshgrid(xs, ys)
    X_fake = np.stack((xx.flatten(), yy.flatten()), axis=1)
    
    # predict by model
    A = model.forward(X_fake.T)
    
    # convert predictions
    if C == 1:
        A = A > threshold
        A = A.reshape(NX, NY)
        Y = np.squeeze(Y)
    else:
        A = A.T
        A = A.argmax(axis=1)
        A = A.reshape(NX, NY)
        Y = Y.argmax(axis=1)
    
    # init plot
    plt.figure()
    plt.title('Decision Map')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.xlim(x_range)
    plt.ylim(y_range)
    
    # plot heatmap
    palette = ('#ef7677ff', '#87b2d4ff', '#94cf92ff', '#c195c8ff', '#ffb266ff', '#ffff85ff', '#ca9a7eff', '#fab3d9ff', '#c2c2c2ff')
    cmap = matplotlib.colors.ListedColormap(palette[:max(C, 2)], N=None)
    plt.contourf(xx, yy, A.astype('int'), cmap=cmap)
    
    # plot contour
    plt.contour(xx, yy, A, colors="#ffffff")
    
    # plot training set
    palette = ('#e41a1cff', '#377eb8ff', '#4daf4aff', '#984ea3ff', '#ff7f00ff', '#ffff33ff', '#a65628ff', '#f781bfff', '#999999ff')
    cmap = matplotlib.colors.ListedColormap(palette[:max(C, 2)], N=None)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=Y, marker='o', cmap=cmap)
    
    # set zoom
    plt.axis([x_range[0], x_range[1], y_range[0], y_range[1]])
    
    # show plot
    plt.show()


def plot_regression(model, X, Y):
    """Plots regression fit."""
    
    # init plot
    plt.figure()
    plt.title('Fit')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # predict by model
    A = model.forward(X)
    
    # plot training set
    plt.scatter(X[0], Y[0], s=40, marker='o', cmap="Blue")
    
    # plot predicted data
    plt.plot(X[0], A[0], 'Red', linewidth=2)
    
    # show plot
    plt.show()
