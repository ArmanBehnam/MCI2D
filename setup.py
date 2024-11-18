from setuptools import setup, find_packages

setup(
    name="mci-progression-analysis",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torch-geometric>=2.3.0',
        'torch-scatter>=2.1.1',
        'torch-sparse>=0.6.17',
        'networkx>=3.1',
        'matplotlib>=3.7.1',
        'numpy>=1.24.3',
        'pandas>=2.0.0',
        'scikit-learn>=1.2.2',
        'python-louvain>=0.16',
    ],
    author="Arman Behnam",
    author_email="abehnam@hawk.iit.edu",
    description="MCI Progression Analysis using Graph Neural Networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ArmanBehnam/MCI2D/tree/main",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "mci-analysis=mci_analysis.main:main",
        ],
    },
)
