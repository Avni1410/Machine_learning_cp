from setuptools import setup, find_packages

setup(
    name="financial_inclusion_africa",
    version="0.1.0",
    author="Kelvin Byabato",
    description="ML pipeline for financial inclusion prediction, explainability, and policy simulation in East Africa.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Add core dependencies
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "shap",
        "optuna"
    ],
    python_requires=">=3.10",
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    url="https://github.com/Byabato/financial-inclusion-africa-ml",
)
