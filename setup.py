from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='rag_pipeline_ops',
    version='0.1.3',
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
    description='A package for RAG Pipeline Operations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3.12'
    ],
)