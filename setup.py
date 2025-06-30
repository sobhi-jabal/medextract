from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="medextract",
    version="0.1.0",
    author="MSJ & EC",
    author_email="m.sobhi.jabal.research@gmail.com",
    description="A tool for extracting clinical datapoints from medical reports using LLMs and RAG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sobhi-jabal/medextract",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "langchain",
        "pyyaml",
        "tqdm",
        "sentence-transformers",
        "faiss-cpu",
        "ollama",
        "interruptingcow",
        "transformers",
        "torch",
    ],
    entry_points={
        "console_scripts": [
            "medextract=medextract:main",
        ],
    },
)