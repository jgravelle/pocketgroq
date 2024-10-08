# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pocketgroq",
    version="0.4.7",  # Incremented the version number to 0.4.6
    author="PocketGroq Team",
    author_email="pocketgroq@example.com",
    description="A library for easy integration with Groq API, including web scraping, image handling, and Chain of Thought reasoning",
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
    packages=find_packages(include=['pocketgroq', 'pocketgroq.*']),
    python_requires=">=3.7",
    install_requires=[
        "bs4>=0.0.2",
        "groq>=0.8.0",
        "python-dotenv>=0.19.1",
        "requests>=2.32.3",
        "langchain>=0.3.1",
        "langchain-groq>=0.2.0",
        "langchain-community>=0.3.1",
        "markdown2>=2.5.0",
        "faiss-cpu>=1.8.0.post1",
        "ollama>=0.3.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "pytest-asyncio>=0.21.0",
        ],
    },
)