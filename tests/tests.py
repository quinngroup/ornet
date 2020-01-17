
import unittest

import test_pipeline

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests([
        loader.loadTestsFromModule(module=test_pipeline)
    ])
    runner = unittest.TextTestRunner()
    runner.run(suite)
        
