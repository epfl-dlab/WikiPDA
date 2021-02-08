"""
A pretty standard setup.py
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="wikipda-lib-DanielBergThomsen",
    version="0.1.0",
    author="Daniel Berg Thomsen",
    author_email="danielbergthomsen@gmail.com",
    description="Library for easy use of WikiPDA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    url="https://github.com/epfl-dlab/wikipda-lib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
