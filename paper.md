---
title: 'OrNet-  a python pipeline to model organelles as social networks'
tags:
  - Python
  - cellular biology
  - mitochondria
  - computer vision
authors:
    - name: Mojtaba Fazli
    - name: Andrew Durden
        affilitation: 2
    - name: Marcus Hill
        affilitation: 2
affiliations:
 - name: Chakra Chennubhotla, Associate Professor, University of Pittsburgh
   index: 1
 - name: Shannon Quinn, Assistant Professor, University of Georgia
   index: 2
date: 19 November 2019
bibliography: paper.bib
---

# Summary

Mitochondria can offer key insights into the health of a human cell. These organelles are strung out across a cell in diffuse, fluctuating webs; healthy structures can be altered by disease, cellular invaders, and other malfunctioning organelles. Fluorescent tagging has shed light on mitochondria’s spatial orientation, but practical observations of structural changes are limited. Biomedical and clinical researchers cannot take advantage of large datasets (i.e. videos of mitochondria in various environments), without automated methods to quantify distribution changes over time. Fitting a model to individual video frames and comparing how the scene evolves has shown some success in measuring spatiotemporal changes [@ruan:2019]. 

Our tool kit builds on this idea, as we seek to convert physical mitochondrial networks into graph networks, then characterize distinct phenotypes. This pipeline is a prototype for future analysis kits. Specifically, it focuses on characterizing two known mitochondrial conditions: drastic fragmentation and excessive fusion. This may assist researchers in larger clinical studies, but expanding this project might eventually yield tools to pick up on structural changes that elude the human eye.

OrNet, a python package, relies heavily on algorithms from the Scikit-Learn tool kit. This in turn is built on NumPy, SciPy, and matplotlib. Our program takes videos of human cells with fluorescently tagged mitochondria and first extracts individual cells. With that base to work from, a model is fitted to video frames by calculating a mixture of probability masses (Gaussian mixture model). Essentially, this model identifies the center of the densest regions of mitochondrial protein, or peaks, and finds the probability that the surrounding protein belongs to a specific local distribution. The ultimate goal is to analyze the data as a graph, where the peaks in density are used as nodes, and probability metrics form weights and edges between nodes. 

The current version requires segmentation masks for each cell in a video, taken from the initial frame of each video, to seed the complete segmentation algorithm. Past masks were generated using the C based ITK-SNAP medical imaging tool. Running the current code requires executing the Pipeline.py script with arguments denoting the location of raw videos, masks, and desired path for output. Output takes the form of learned components from the mixture model and the calculated weights between nodes. 

# Acknowledgements

# License

# References
