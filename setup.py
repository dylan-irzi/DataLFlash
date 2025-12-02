from setuptools import setup, find_packages

setup(
    name="datalflash",
    version="0.1.1",
    author="Juan Castro",
    author_email="dylan.irzi@gmail.com",
    description="High-performance data loaders for PyTorch inspired by flashtensors",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dylanirzi/DataLFlash",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "tqdm>=4.64.0",
        "psutil>=5.9.0",
        "pyarrow>=8.0.0",
        "zarr>=2.13.0",
        "dask>=2022.10.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
)