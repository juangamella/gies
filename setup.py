import setuptools

setuptools.setup(
    name="gies",
    version="0.0.2",
    author="Juan L. Gamella, Olga Kolotuhina",
    author_email="juangamella@gmail.com",
    packages=["gies", "gies.test", "gies.scores"],
    scripts=[],
    url="https://github.com/juangamella/gies",
    license="BSD 3-Clause License",
    description="Python implementation of the GIES algorithm for causal discovery",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=["numpy>=1.15.0"],
)
