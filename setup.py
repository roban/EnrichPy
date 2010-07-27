#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
import os
import nose

setup(
    name = "EnrichPy",
    version = "0.1",
    packages = find_packages(),
    package_data = {
        # If any package contains these files, include them
        '': ['*.dat', 'yield_data/*.dat','yield_data/*/*.dat'],
        },
    install_requires = ['numpy', 'scipy', 'cosmolopy'],

    tests_require = ['nose',],
    test_suite = 'nose.collector',

    # metadata for upload to PyPI
    author = "Roban Hultman Kramer",
    author_email = "robanhk@gmail.com",
    description = "a package of routines related to chemical enrichment",
#    url = "http://roban.github.com/EnrichPy/",   # project home page
#    keywords = ("astronomy cosmology cosmological distance density galaxy" +
#                "luminosity magnitude reionization Press-Schechter Schecter"),
    license = "MIT",
#    long_description = \
#    """ """,
    classifiers = ['License :: OSI Approved :: MIT License',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 2.6',
                   'Topic :: Scientific/Engineering :: Astronomy',
                   'Operating System :: OS Independent'
                   ]
)
