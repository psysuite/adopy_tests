"""Setup script for postprocessing package."""

from setuptools import setup, find_packages

setup(
    name="postprocessing",
    version="1.0.0",
    description="MATLAB to Python postprocessing conversion for temporal bisection data",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
        "openpyxl>=3.0.0",
    ],
    extras_require={
        "test": [
            "hypothesis>=6.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
)
