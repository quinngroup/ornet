'''
'''

import os
import sys
import argparse

import cv2
import imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

from ornet.gmm.image import img_to_px
from ornet.analysis.util import sort_eigens

def normalize_colors(matrix, max_intensity=255):
    '''
    Normalize a matrix to a range of colors.

    Parameters
    ----------
    matrix: matrix (N x M)
        Input matrix to be normalized. Can also be a vector.
    max_intensity: int
        The range of color values will be constrained to 
        [0, max_intensity]. Default is 255.

    Returns
    -------
    scaled_matrix: matrix (N x M)
        A matrix whose values are scaled in the range of
        [0, max_intensity].
    '''

    colors = []
    min_element = np.min(matrix)
    max_element = np.max(matrix)
    diff = max_element - min_element
    return (((matrix - min_element) / diff) * max_intensity).astype(np.uint8)

def region_flow(inter_path, eigen_path, vid_path, outdir, cmap='tab20b'):
    '''
    '''

    out_vid_path = os.path.join(outdir, 'output.avi')
    inter = np.load(inter_path)
    means, covars, weights, precs = \
        inter['means'], inter['covars'], inter['weights'], inter['precs']

    eigens = np.load(eigen_path)
    eigen_vals, eigen_vecs = eigens['eigen_vals'], eigens['eigen_vecs']

    with imageio.get_reader(vid_path) as reader:
        metadata = reader.get_meta_data()
        vid = np.array(list(reader))
        fps = metadata['fps']
        size = metadata['size']

    with imageio.get_writer(out_vid_path, mode='I', fps=1) as writer:
        for i in tqdm(range(len(vid))):
            gmm = GaussianMixture(
                n_components=covars[i].shape[0],
                means_init=means[i],
                weights_init=weights[i],
                precisions_init=precs[i]
            )
            X = img_to_px(vid[i,:,:,0])
            gmm.fit(X)
            pred = gmm.predict_proba(X)
            colors = normalize_colors(eigen_vecs[i,:,0])

            out_frame = np.zeros(size)
            img_coordinates = []
            for j, row in enumerate(X):
                row_list = row.tolist()
                if row_list not in img_coordinates:
                    pixel_intensity = 0
                    for k, weight in enumerate(pred[j]):
                        pixel_intensity += colors[k] * weight

                    out_frame[row[0], row[1]] = int(pixel_intensity)
                    img_coordinates.append(row_list)

            ax = plt.imshow(out_frame, cmap=cmap)
            mapped_frame = ax.make_image('AGG', unsampled=True)[0][:,:,:3]
            writer.append_data(mapped_frame)

            '''
            #Current video Frame and the color mapped variant
            ax = plt.subplot(121)
            img = ax.imshow(vid[i,:,:,0])
            ax.set_title('Single Video Frame')
            plt.colorbar(img, ax=ax)
            ax = plt.subplot(122)
            img = ax.imshow(mapped_frame)
            ax.set_title('Color Mapped Frame')
            plt.colorbar(img, ax=ax)
            '''

            '''
            #Mixture Component Ellipses
            fig = plt.figure()
            ax = fig.add_subplot(121) for j, mean in enumerate(means[i]):
                e = Ellipse(
                        (mean[0], mean[1]),
                        width=((mean[0] + covars[i, j, 0, 0] * 3) 
                               - (mean[0] - covars[i, j, 0, 0] * 3)),
                        height=((mean[1] + covars[i, j, 1, 1] * 3) 
                               - (mean[1] - covars[i, j, 1, 1] * 3))
                )
                ax.add_artist(e)
                #e.set_facecolor(colors[j])
                e.set_alpha(np.random.rand())
                e.set_facecolor(np.random.rand(3))


            ax = fig.add_subplot(122)
            ax.imshow(out_frame, cmap=cmap)
            #plt.colorbar()
            plt.show()
            '''

def parse_cli(args):
    parser = argparse.ArgumentParser(
                description='Visualizes organellar regions.')
    parser.add_argument('-i', '--intermediates', required=True,
                        help='Path to intermediates file (.npz).')
    parser.add_argument('-e', '--eigens', required=True,
                        help='Path to eigen file (.npz).')
    parser.add_argument('-v', '--video', required=True,
                        help='Path to grayscale cell video (.avi).')
    parser.add_argument('-o', '--outdir', default=os.getcwd(),
                        help='Path to output directory.')
    return vars(parser.parse_args(args))

def main():
    args = parse_cli(sys.argv[1:])
    region_flow(args['intermediates'], args['eigens'],
                args['video'], args['outdir'], cmap='Blues') #cmap='jet'

if __name__ == '__main__':
    main()
