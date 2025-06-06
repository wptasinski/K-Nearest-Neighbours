import numpy as np
import pdb
import matplotlib.pyplot as plt
from matplotlib import style
import itertools
import math
from collections import Counter
import KNN_visualizations as KNNV
style.use('fivethirtyeight')



def main():
    """
    Runs the interactive K-Nearest Neighbors classification and visualization.
    
    Initializes a labeled 2D dataset, displays it on a plot, and allows the user to input points interactively by clicking. Each input point is classified using the KNN algorithm, visualized with connections to its nearest neighbors, and added to the dataset. The process continues until all user-input points are classified and displayed.
    """
    data = {
        'y':[[1,1],[1,2],[2,1],[1,0]],
        'b':[[5,4],[5,5],[4,5],[6,1]],
        'r':[[1,4],[1,3],[2,5],[3,4]],
        }
    k=7
    fig = plt.figure()
    # Initial display poitns from given dataset
    KNNV.scatter_data_points(data)
    # Values to input by clicking on the plot
    num_of_input_values = 10
    y = plt.ginput(num_of_input_values)

    # Run KNN Algorithm
    K_Nearest(data, y, k, fig)
    plt.show()
    
def K_Nearest(data, X, k, fig):
    for x in X:
        # List without shape value on [1] can throw error
        try:
            # Checking how many points is in the X
            if np.shape(X)[1]>0:
                pass
        except Exception as e:
            x=X

        # Update plot info
        plt.text(0.5, 5.5, 'K='+str(k))
        plt.text(1, 5.5, 'Actual point: [%d,%d]'%(x[0],x[1]))

        # Display all dataset points and tested point
        plt.scatter(x[0],x[1],c='g',s=100)
        KNNV.scatter_data_points(data)

        # We can mix values because one vector from all of the values stands for
        # one row of data in real dataframe so it's unique and has got it's own label
        dataList = list(itertools.chain.from_iterable(data.values()))

        # Drawing all connections with current point
        KNNV.draw_pt_connections(x,dataList)

        # Counting euclidean distances from current
        # tested point to the rest of points
        euclidList = sorted(dataList, key=lambda pt: euclidean_dist(x, pt))

        # Assigning tested point to the group with the biggest
        # frequency of appearance
        group = assign_to_group(x, euclidList[:k], data, k)

        # Clearing figure
        plt.clf()
        KNNV.scatter_data_points(data)
        plt.scatter(x[0],x[1],c='g',s=100)

        # Plotting only the k nearest euclidean distances
        KNNV.draw_pt_connections(x,euclidList[:k])
        plt.scatter(x[0],x[1],c=group,s=100)

        # Assigning current point to the new group
        data[group].append(x)

        # Updating figure with new grouped data
        plt.clf()
        KNNV.scatter_data_points(data)

def assign_to_group(pt, euclidList, data, k):
    """
        Assigns given point to the group with highest frequency of K points
    """
    votes = []
    for g in euclidList[:k]:
        for key, points in data.items():
            for pt in points:
                if pt==g:
                    votes.append(key)
    return max(set(votes), key = votes.count)

def euclidean_dist(point, pt):
    sum = 0
    for q,p in zip(point, pt):
        sum+=(q-p)**2
    return math.sqrt(sum)

if __name__ == '__main__':
    main()
