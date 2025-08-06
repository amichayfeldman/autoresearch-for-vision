"""Setup script for CV Error Analyzer."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cv-error-analyzer",
    version="0.1.0",
    author="CV Error Analysis Team",
    description="Automated agentic AI system for computer vision model error analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "albumentations>=1.3.0",
        "ultralytics>=8.0.0",
        "clearml>=1.13.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "pydantic>=1.10.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cv-error-analyzer=cv_error_analyzer.cli:main",
        ],
    },
)