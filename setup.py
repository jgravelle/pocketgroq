from setuptools import setup, find_packages

setup(
    name="pocketgroq",
    version="0.1.5",
    author="PocketGroq Team",
    author_email="pocketgroq@example.com",
    description="A library for easy integration with Groq API",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jgravelle/pocketgroq",
    packages=find_packages(),
    install_requires=[
        "groq==0.8.0",
        "python-dotenv==0.19.1",
    ],
    extras_require={
        "dev": [
            "pytest==7.3.1",
            "pytest-asyncio==0.21.0",
        ],
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
    python_requires=">=3.7",
)