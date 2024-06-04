from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='bombay',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'hnswlib',
        'openai',
        'pytest',
        'chromadb',
    ],
    author='faith6',
    author_email='root39293@gmail.com',
    description='A package for building RAG-based LLM pipelines',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3.12'
    ],
)