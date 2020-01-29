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
  - name: Rachel Mattson
    affiliation: 1 
  - name: Chakra Chennubhotla
    affiliation: 2
  - name: Shannon Quinn
    orcid: 0000-0002-8916-6335
    affiliation: 1, 3
affiliations:
  - name: Department of Computer Science, University of Georgia, Athens, GA 30602 USA
    index: 1
  - name: Department of Computational and Systems Biology, University of Pittsburgh, Pittsburgh, PA 15232 USA
    index: 2 
  - name: Department of Cellular Biology, University of Georgia, Athens, GA 30602 USA
    index: 3
date: 19 November 2019
bibliography: paper.bib
---

# Summary

Mitochondria can offer key insights into the health of a human cell. These organelles are strung out across a cell in diffuse, fluctuating webs; healthy structures can be altered by disease, cellular invaders, and other malfunctioning organelles [@Stavru:2016]. Previous work using mitochondrial distribution focuses on building a model of the cell’s morphology. For instance, both the Allen Cell Structure Segmenter [@Chen:2016] and CellOrganizer [@Murphy:2015] toolkits generate models for organelle structure, providing visual or probabilistic representations. Successful generation of static models has led to investigations in how structures develop over time, such as using mitochondrial distributions to map cells as they grow into a specialized form [@Ruan:2019].  This describes a single, albeit complex cellular process; many more processes could be described by mitochondrial distributions. 

Our project builds on this idea [@Durden:2019], as we seek to convert physical mitochondrial networks into graph networks, then characterize distinct phenotypes. This tool kit is a prototype for future analysis kits. As it stands, we focus on characterizing two known mitochondrial conditions: drastic fragmentation and excessive fusion. Our resulting model seeks to pull out the key features that make up these processes from a data set, rather than immediately classify the set. This creates a staging ground for biologists and clinicians to use our pipeline to examine evolving mitochondrial distributions in more detail.

OrNet, a python package [@Python:1995], relies heavily on algorithms from the Scikit-Learn tool kit [@scikit-learn:2011]. This in turn is built on NumPy [@Numpy:2006], SciPy [@2019arXiv190710121V], and Matplotlib [@Hunter:2007]. Our program takes videos of human cells with fluorescently tagged mitochondria and first extracts individual cells. With that base to work from, a model is fitted to video frames by calculating a mixture of probability masses (Gaussian Mixture Model). Essentially, this model identifies the center of the densest regions of mitochondrial protein, or peaks, and finds the probability that the surrounding protein belongs to a specific local distribution. The ultimate goal is to analyze the data as a graph, where the peaks in density are used as nodes, and probability metrics form weights and edges between nodes.

The current version requires segmentation masks for each cell in a video, taken from the initial frame of each video, to seed the complete segmentation algorithm. Past masks were generated using the C based ITK-SNAP medical imaging tool [@Yoo:2002]. Running the current code requires executing the Pipeline.py script with arguments denoting the location of raw videos, masks, and desired path for output. Output takes the form of learned components from the mixture model and the calculated weights between nodes. 


# Acknowledgements

Thanks to Allyson T. Loy, Barbara Reaves, Abigail Courtney, and Frederick D. Quinn for contributions to the associated project.

The project that yielded this software was supported in part by a grant from the National Science Foundation (#1458766).

# References

---
nocite: |
    @behnel:2010, @Clark:2016, 
    @Hagberg:2008, @kiwisolver2018, 
    @Klein:2019, @Lee:2019,
    @Mcguire:2007, @opencv_library,
    @scikit-image:2014, @Andrew_Durden-proc-scipy-2018,
    @Costa-Luis:2019
---
