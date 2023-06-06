# Transformer playground

Test playground for transformer projects. Main models support. Decoder, encoder and encoder-decoder based transformers.

## Setup

The best way to use is to setup a conda environment for the project and install it.

For training:

```bash
conda env create -f requirements/conda-environment.yaml && conda clean -q -y -a 
```

## Use docker

Build image:

```bash
pushd scripts && chmod +x build_image.sh && build_image.sh && popd
```

Set mapped volumes in scripts/build_container.sh and use it to create container:

```bash
pushd scripts && chmod +x run_container.sh && run_container.sh && popd
```
