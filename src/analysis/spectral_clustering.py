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

def main():
    '''
    '''

    k = 3
    outdir_path = '/home/marcus/Desktop/Spectral-Clustering-Plots/'
    eigen_data = np.load(
        '/extrastorage/ornet/Eigenspectrum/Eigendata/DsRed2-HeLa_3_1_LLO_1.npz'
    )
    eigen_vals, eigen_vecs = eigen_data['eigen_vals'], eigen_data['eigen_vecs']
    sns.set()
    colors = {}
    for i in range(k):
        colors[i] = np.random.rand(3)

    for i in tqdm(range(eigen_vecs.shape[0])): 
        kmeans = KMeans(n_clusters=k).fit(eigen_vecs[i])
        label_colors = [colors[x] for x in kmeans.labels_]
        
        fig = plt.figure(figsize=(19.2, 9.60))
        ax = fig.add_subplot(111)
        ax.scatter(eigen_vecs[i,:,0], eigen_vecs[i,:,1], c=label_colors)
        ax.set_title('DsRed2-HeLa_3_1_LLO_1')
        ax.set_xlabel('First Dimension')
        ax.set_ylabel('Second Dimension')
        for j in range(len(label_colors)):
            ax.annotate(str(j), (eigen_vecs[i,j,0], eigen_vecs[i,j,1]))
        plt.savefig(os.path.join(outdir_path, str(i)))
        #plt.show()
        plt.close()

if __name__ == '__main__':
    main()
