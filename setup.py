# -*- coding: utf-8 -*-
import os

from setuptools import setup, find_packages


def filepath(fname):
    return os.path.join(os.path.dirname(__file__), fname)


with open("reqs/base-requirements.txt") as f:
    requirements = f.read().splitlines()


exec(compile(open("lifelike/version.py").read(), "lifelike/version.py", "exec"))

with open("README.md") as f:
    long_description = f.read()

setup(
    name="lifelike",
    version=__version__,
    author="Cameron Davidson-Pilon",
    author_email="cam.davidson.pilon@gmail.com",
    description="Survival analysis prediction in Python",
    license="MIT",
    keywords="survival analysis prediction jax deep learning",
    url="https://github.com/CamDavidsonPilon/lifelike",
    packages=find_packages(),
    python_requires=">=3.5",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=requirements,
    package_data={"lifelike": ["../README.md", "../LICENSE"]},
)
