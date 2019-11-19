---
title: 'OrNet - a Python Toolkit to Model the Diffuse Structure of Organelles as Social Networks'
tags:
  - Python
  - cellular biology
  - mitochondria
  - computer vision
authors:
  - name: Mojtaba Fazli
    orcid: 0000-0002-6082-2538
    affiliation: 1
  - name: Andrew Durden
    affiliation: 1
  - name: Marcus Hill
    orcid: 0000-0002-9380-3181
    affiliation: 1
  - name: Shannon P. Quinn
    orcid: 0000-0002-8916-6335
    affiliation: 1
  - name: Chakra Chennubhotla
    affiliation: 2
  - name: Rachel Mattson
    affiliation: 2
  - name: Chakra Chennubhotla
    index: 1
  - name: Shannon Quinn
    index: 2, 3
affiliations:
  - name: Department of Computational and Systems Biology, University of Pittsburgh, Pittsburgh, PA 15232 USA
    index: 1
  - name: Department of Computer Science, University of Georgia, Athens, GA 30602 USA
    index: 2
  - name: Department of Cellular Biology, University of Georgia, Athens, GA 30602 USA
    index: 3
date: 19 November 2019
bibliography: paper.bib
---

# Summary

Mitochondria can offer key insights into the health of a human cell. These organelles are strung out across a cell in diffuse, fluctuating webs; healthy structures can be altered by disease, cellular invaders, and other malfunctioning organelles. Fluorescent tagging has shed light on mitochondria’s spatial orientation, but practical observations of structural changes are limited. Biomedical and clinical researchers cannot take advantage of large datasets (i.e. videos of mitochondria in various environments), without automated methods to quantify distribution changes over time. Fitting a model to individual video frames and comparing how the scene evolves has shown some success in measuring spatiotemporal changes (Ruan, 2019). 

Our project builds on this idea (see Durden, 2019), as we seek to convert physical mitochondrial networks into graph networks, then characterize distinct phenotypes. This tool kit is a prototype for future analysis kits. Specifically, it focuses on characterizing two known mitochondrial conditions: drastic fragmentation and excessive fusion. This may assist researchers in larger clinical studies, but expanding this project might eventually yield tools to pick up on structural changes that elude the human eye.

OrNet, a python package, relies heavily on algorithms from the Scikit-Learn tool kit. This in turn is built on NumPy, SciPy, and Matplotlib. Our program takes videos of human cells with fluorescently tagged mitochondria and first extracts individual cells. With that base to work from, a model is fitted to video frames by calculating a mixture of probability masses (Gaussian Mixture Model). Essentially, this model identifies the center of the densest regions of mitochondrial protein, or peaks, and finds the probability that the surrounding protein belongs to a specific local distribution. The ultimate goal is to analyze the data as a graph, where the peaks in density are used as nodes, and probability metrics form weights and edges between nodes. 

The current version requires segmentation masks for each cell in a video, taken from the initial frame of each video, to seed the complete segmentation algorithm. Past masks were generated using the C based ITK-SNAP medical imaging tool. Running the current code requires executing the Pipeline.py script with arguments denoting the location of raw videos, masks, and desired path for output. Output takes the form of learned components from the mixture model and the calculated weights between nodes. These outputs are ripe for analysis at the researcher's discretion.

# Acknowledgements

Thanks to Allyson T. Loy, Barbara Reaves, Abigail Courtney, and Frederick D. Quinn for contributions to the associated project.

The project that yielded this software was supported in part by a grant from the National Science Foundation (#1458766).

# References
