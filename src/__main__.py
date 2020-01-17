'''
Interface to run the pipeline from the command line.
'''

import os
import sys
import argparse

import ornet.Pipeline as pipeline

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
    parser.add_argument('-o', '--output',
                        help='Output directory to save files.',
                        default=os.getcwd())
    return vars(parser.parse_args(args))


def main(system_args):
    args = parse_cli(system_args[1:])
    pipeline.run(args['input'], args['masks'], args['output'])

if __name__ == '__main__':
    main(sys.argv)
