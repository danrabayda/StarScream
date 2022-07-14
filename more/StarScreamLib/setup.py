from setuptools import find_packages, setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='StarScreamLib',
    packages=find_packages(include=['StarScreamLib']),
    version='0.1.2',
    description='A library for working with aircraft audios',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Daniel Rabayda',
    author_email='rabaydadp@gmail.com',
    license='MIT',
    url='https://github.com/danrabayda/StarScream/tree/main/more/StarScreamLib',
    install_requires=[
        'pydub',
        'ipywidgets',
        'numpy',
        'scipy',
        'matplotlib',
        'IPython',
      ],
)
