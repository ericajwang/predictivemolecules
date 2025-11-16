from setuptools import setup, find_packages

setup(
    name="predictivemolecules",
    version="1.0.0",
    description="Deep learning framework for small molecule binding prediction",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "torch-geometric>=2.0.0",
        "rdkit-pypi>=2022.3.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "tensorboard>=2.8.0",
        "joblib>=1.1.0",
    ],
    python_requires=">=3.8",
)

