from setuptools import setup, find_packages

setup(
  name = 'se3-transformer-pytorch',
  packages = find_packages(),
  include_package_data = True,
  version = '0.2.3',
  license='MIT',
  description = 'SE3 Transformer - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/se3-transformer-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'equivariance',
    'SE3'
  ],
  install_requires=[
    'einops>=0.3',
    'filelock',
    'numpy',
    'torch>=1.6'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest',
    'lie_learn',
    'numpy',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
