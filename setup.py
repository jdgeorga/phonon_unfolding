from setuptools import setup, find_packages

setup(
    name='phonon_unfolding',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for unfolding phonon band structures and parallel force calculations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/phonon_unfolding',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'ase',
        'phonopy',
        'mpi4py'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'calculate-forces=phonon_unfolding.scripts.calculate_forces:main',
        ],
    },
)