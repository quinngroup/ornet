'''
Interface to run the pipeline from the command line.
'''

import os
import sys
import argparse

import ornet.pipeline as pipeline

def parse_cli(args):
    '''
    Parses arguments from the cli.

    Parameters
    ----------
    args: list of strings
        Argument options and values.

    Returns
    ----------
    args: dict
        Argument options are the keys and the values are
        the information supplied from the cli.
    '''

    parser = argparse.ArgumentParser(
        usage = 'python -m ornet [-h] -i INPUT -m MASKS -o OUTPUT',
        description='An end-to-end pipeline of OrNet.'
    )
    parser.add_argument('-i', '--input',
                        help='Input directory containing video(s).',
                        required=True)
    parser.add_argument('-m', '--masks',
                        help='Input directory containing vtk mask(s).',
                        required=True)
    parser.add_argument('-o', '--output', default=os.getcwd(),
                        help='Output directory to save files.')
    parser.add_argument('-c', '--count', type=int, default=-1,
                        help='First N frames of video to use. Default is all.')
    return vars(parser.parse_args(args))


def main(system_args):
    args = parse_cli(system_args[1:])
    pipeline.run(args['input'], args['masks'], args['output'], args['count'])

if __name__ == '__main__':
    main(sys.argv)
