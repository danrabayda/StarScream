from setuptools import find_packages, setup
setup(
    name='StarScreamLib',
    packages=find_packages(include=['StarScreamLib']),
    version='0.1.0',
    description='A library for working with aircraft audios',
    author='Daniel Rabayda',
    author_email='rabaydadp@gmail.com',
    license='MIT',
    url='https://github.com/danrabayda/StarScream/tree/main/more/StarScreamLib',
    install_requires=[
        'pydub',
        'ipywidgets',
        'numpy',
        'io',
        'os',
        'scipy',
        'matplotlib',
        'IPython',
      ],
)
