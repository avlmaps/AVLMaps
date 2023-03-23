from pathlib import Path
from setuptools import setup

description = ['Audio Visual Language Maps for Robot Navigation']

root = Path(__file__).parent
with open(str(root / 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()
version = "0.0"
# with open(str(root / 'requirements.txt'), 'r') as f:
#     dependencies = f.read().split('\n')

setup(
    name='avlmaps',
    version=version,
    packages=['lseg', 'audioclip', 'avlmaps_utils', 'vlmaps'],
    python_requires='>=3.8',
    # install_requires=dependencies,
    author='Chenguang Huang',
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/avlmaps/AVLMaps',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
