import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="map_sims",
    version="0.0.1",
    author="Stijn Debackere",
    author_email="debackere@strw.leidenuniv.nl",
    description="A package to slice simulation particle data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=['map_sims'],
    entry_points={
        "console_scripts": [
            "dagster_map_sims = map_sims.maps.scripts.map_sims_dagster_cli:main",
            "extract_masses = map_sims.extraction.scripts.extract_masses_cli:main",
            "join_mass_files = map_sims.extraction.scripts.join_mass_files_cli:main",
            "map_sims = map_sims.maps.scripts.map_sims_cli:main",
            "slurm_extract_masses = map_sims.extraction.scripts.extract_masses_slurm_run:main",
            "slurm_map_sims = map_sims.maps.scripts.map_sims_slurm_run:main",
        ],
    },
    install_requires=[
        "astropy>=5.0",
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
