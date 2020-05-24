import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="molnet-python", # Replace with your own username
    version="0.0.1",
    author="Minys",
    author_email="minys@foxmail.com",
    description="A small package for loading data from MoleculeNet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Minys233/molnet-python.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: Console"
    ],
    python_requires='>=3.6',
)