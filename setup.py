from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    install_requires = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="bangalore-house-price-prediction",
    version="1.0.0",
    author="Ramyasrivalli",
    author_email="ramyasrivalli@example.com",
    description="Production-grade ML system for Bangalore house price prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ramyasrivalli/Bangalore-House-Price-Prediction",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "bhp-train=train:main",
        ]
    },
)
