#!/bin/bash
pip install -r requirements.txt
pip install -r avlmaps/audioclip/requirements.txt

conda install habitat-sim=0.2.2 -c conda-forge -c aihabitat -y

pip install -e .

cd ~
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e .
