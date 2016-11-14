---
title: Region of Interests
layout: documentation
toc-section: 3
toc-subsection: 2
---

![left30]({{site.baseurl}}/images/components/roi/roi.png)

Segmenting large volumes can be tedious and time consuming. Region of Interests (RoIs) help to focus in specific areas of the volume while limiting every action under SuRVoS to the selected area. By doing so, computed features could be enhanced by reducing noise present in some specific regions of the volume, super-regions (both supervoxels and megavoxels) would more accurately represent the underlying data and predictions of a trained volume can be limited to the area of interest.

RoIs are computed by inserting a bounding cube coordinates (e.g. slices 25 to 50 in Z axis and full length in Y and X) and clicking on **Add**. The default RoI (the only one that cannot be deleted) refeers to the whole volume and by default, everything applyed under SuRVoS will be done on the whole data. By selecting one of the computed RoIs, the visualization will immediately change to only show the selected RoI.

Another very interesting feature of these RoI is that by limiting every method of SuRVoS to the selected area, it not only improves the efficiency and computation time (several orders of magnitude if the RoI is small) but it also allows to segment large volumes in a standard laptop by exploiting RoI + HDF5 data files. RoI limit memory usage and computational complexity to an smaller area of the image while the underlying HDF5 system loads only the required data onto memory, making it feasible to segment large volumes as long as the selected RoI fits into the GPU memory.
