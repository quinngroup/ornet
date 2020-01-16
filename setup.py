
import setuptools

setuptools.setup(
	name='ornet',
	version='0.1',
	packages=['ornet', 'ornet.gmm'],
	package_dir={'ornet':'src', 'ornet.gmm':'src/gmm'},
	entry_points={'console_scripts': ['ornet = ornet.Pipeline:main']},
	install_requires=[
		'itk',
		'numpy',
		'scipy',
		'cython',
		'joblib',
		'imageio',
		'matplotlib',
		'scikit-image',
		'scikit-learn',
		'imageio-ffmpeg',
		'opencv-python>=4.0.0'
	]
)
