---
title: 'OrNet - a Python Toolkit to Model the Diffuse Structure of Organelles as Social Networks'
tags:
  - Python
  - Cellular Biology
  - Organelles 
  - Computer Vision
authors:
  - name: Mojtaba Fazli
    orcid: 0000-0002-6082-2538
    affiliation: 1,*
  - name: Marcus Hill
    orcid: 0000-0002-9380-3181
    affiliation: 1,*
  - name: Andrew Durden
    affiliation: 1
  - name: Rachel Mattson
    affiliation: 1 
  - name: Chakra Chennubhotla
    affiliation: 2
  - name: Shannon Quinn
    orcid: 0000-0002-8916-6335
    affiliation: 1,3
affiliations:
  - name: Department of Computer Science, University of Georgia, Athens, GA 30602 USA
    index: 1
  - name: Department of Computational and Systems Biology, University of Pittsburgh, Pittsburgh, PA 15232 USA
    index: 2 
  - name: Department of Cellular Biology, University of Georgia, Athens, GA 30602 USA
    index: 3
  - name: The two first authors made equal contributions.
    index: *
date: 19 November 2019
bibliography: paper.bib
---

# Summary

Fluorescent microscopy videos are vital for analyzing the morphological changes that subcellular protein structures undergo after exposure to external stimuli. Changes in organelle structures offer crucial insight into the manner in which cells respond to viral or bacterial infections, cellular invaders, or even the organelles themselves malfunctioning [@Stavru:2011]. Generally, modeling organellar structures involve manually inspecting each video then denoting time points and regions that demonstrate anomalous behavior. However, manual techniques are error prone and not scalable to high volumes of data. Thus, arises the need to find a methodology that generates quantitative models capable of accurately describing the data [@Eliceiri:2012]. Prior works have demonstrated successful generation of static models [@Murphy:2015; @Chen:2016; @Ruan:2019]. Such advancements have inspired us to propose a novel framework, OrNet, that models both the spatial and temporal morphology changes that organelles undergo as dynamic social networks.

OrNet is an open-source python package [@Python:1995] that is built-upon the libraries of Scikit-Learn [@scikit-learn:2011], NumPy [@Numpy:2006], SciPy [@2019arXiv190710121V], and Matplotlib [@Hunter:2007]. Our tool accepts as input microscopy videos of cells with fluorescently tagged organelles, and outputs quantitative descriptions of the morphological changes. The methodology consists of developing social networks that capture the spatio-temporal relationships of organellar structures via a dynamic edge management process. The graphs are constructed by fitting gaussian mixture models to every frame of an input video; the final means become the vertices, while a divergence metric is applied to every combination pair of mixture components to create the edges. Once graphs are created for each frame, spectral decomposition is utilized to track the leading eigenvalues to understand the time-points and frame regions where organellar structures are demonstrating significant changes. 

The viability of OrNet has been illustrated by [@Durden:2019] when the framework was utilized to model mitochondria found in HeLa cells that were exposed to various stimuli. We hope that our tool will be utilized by any project seeking to model subcellular organelles. 


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
