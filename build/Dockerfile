FROM continuumio/miniconda3

WORKDIR /developer

# Create the environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate the environment
RUN conda activate fuzzy

