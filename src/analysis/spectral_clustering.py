'''
Clusters eigenvalue row vectors, because they correspond
with guassian mixture compoonents, to understand spatial
regions that are demonstrating similar behaviors.
'''

import os

import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

def main():
    '''
    '''

    k = 3
    #outdir_path = '/home/marcus/Desktop/Spectral-Clustering-Plots/'
    outdir_path = '/home/marcus/Desktop/Spectral-Clustering-Plots/'
    os.makedirs(outdir_path, exist_ok=True)
    eigen_data = np.load(
        #'/extrastorage/ornet/Eigenspectrum/Eigendata/DsRed2-HeLa_3_1_LLO_1.npz'
        '/extrastorage/ornet/Eigenspectrum/Eigendata/DsRed2-HeLa_3_15_LLO1part1_1.npz'
    )
    eigen_vals, eigen_vecs = eigen_data['eigen_vals'], eigen_data['eigen_vecs']
    sns.set()
    plot_in_3D = False
    colors={}
    colors[0], colors[1], colors[2] = 'r', 'g', 'b'
    '''
    colors = {}
    for i in range(k):
        colors[i] = np.random.rand(3)
    '''

    for i in tqdm(range(eigen_vecs.shape[0])): 
        kmeans = KMeans(n_clusters=k).fit(eigen_vecs[i, :, :k])
        label_colors = [colors[x] for x in kmeans.labels_]
        
        fig = plt.figure(figsize=(19.2, 9.60))
        if plot_in_3D:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(eigen_vecs[i,:,0], eigen_vecs[i,:,1], c=label_colors)
            ax.set_zlabel('Third Dimension')
        else:
            ax = fig.add_subplot(111)
            ax.scatter(eigen_vecs[i,:,0], eigen_vecs[i,:,1], c=label_colors)
            #ax.scatter(eigen_vecs[i,:,-2], eigen_vecs[i,:,-3], c=label_colors)

        #ax.set_title('DsRed2-HeLa_3_1_LLO_1')
        ax.set_title('DsRed2-HeLa_3_15_LLO1part1_1')
        ax.set_xlabel('First Dimension')
        ax.set_ylabel('Second Dimension')
        ax.set_xlim(left=-.65, right=.65)
        ax.set_ylim(bottom=-.6, top=.6)
        for j in range(len(label_colors)):
            if plot_in_3D:
                ax.text(eigen_vecs[i,j,0], eigen_vecs[i,j,1], 
                        eigen_vecs[i,j,2], str(j))
            else:
                ax.text(eigen_vecs[i,j,0], eigen_vecs[i,j,1], str(j))
                #ax.text(eigen_vecs[i,j,-2], eigen_vecs[i,j,-3], str(j))

        plt.savefig(os.path.join(outdir_path, str(i)))
        #plt.show()
        plt.close()

if __name__ == '__main__':
    main()
