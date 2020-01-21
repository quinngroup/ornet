
import os
import unittest

import test_pipeline

if __name__ == '__main__':
    os.environ['IMAGEIO_FFMPEG_EXE'] = 'ffmpeg'
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests([
        loader.loadTestsFromModule(module=test_pipeline)
    ])
    runner = unittest.TextTestRunner(warnings='ignore')
    runner.run(suite)
        
