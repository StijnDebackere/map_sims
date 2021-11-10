import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simulation_slices",
    version="0.0.1",
    author="Stijn Debackere",
    author_email="debackere@strw.leidenuniv.nl",
    description="A package to slice simulation particle data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=['simulation_slices'],
    install_requires=[
        "astropy",
        "dagster",
        "h5py",
        # "gadget @ https://github.com/StijnDebackere/gadget",
        # "mira_titan @ https://github.com/StijnDebackere/mira_titan",
        "numba",
        "numpy",
        "scipy",
        "toml",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
