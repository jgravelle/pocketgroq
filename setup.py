from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pocketgroq",
    version="0.2.4",  # Increment the version number
    author="PocketGroq Team",
    author_email="pocketgroq@example.com",
    description="A library for easy integration with Groq API, including image handling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jgravelle/pocketgroq",
    project_urls={
        "Bug Tracker": "https://github.com/jgravelle/pocketgroq/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.7",
    install_requires=[
        "groq==0.8.0",
        "python-dotenv==0.19.1",
        "requests>=2.32.3",
    ],
    extras_require={
        "dev": [
            "pytest==7.3.1",
            "pytest-asyncio==0.21.0",
        ],
    },
)