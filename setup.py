#setup.py
from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='bombay',
    version='0.1.2',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'hnswlib',
        'openai',
        'chromadb',
        'termcolor',
        'colorama',
    ],
    author='faith6',
    author_email='root39293@gmail.com',
    description='A package for building RAG-based LLM pipelines',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3.12'
    ],
    entry_points={
        'console_scripts': [
            'bombay=bombay.bombay_cli:main',
        ],
    },
)