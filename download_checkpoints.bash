#!/bin/bash
# download AudioCLIP checkpoints
mkdir avlmaps/audioclip/assets
wget https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Partial-Training.pt -P avlmaps/audioclip/assets/
wget https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/bpe_simple_vocab_16e6.txt.gz -P avlmaps/audioclip/assets/

# download LSeg checkpoints
mkdir avlmaps/lseg/checkpoints
cd avlmaps/lseg/checkpoints
gdown --fuzzy https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view
cd ../../..