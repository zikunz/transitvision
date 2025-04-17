from setuptools import setup, find_packages

setup(
    name="transitvision",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "plotly",
    ],
    extras_require={
        "ml": ["tensorflow", "torch", "transformers"],
        "geo": ["geopandas", "rasterio", "pyproj"],
        "dev": ["pytest", "pytest-cov", "black", "flake8", "mypy"],
    },
    python_requires=">=3.8",
)