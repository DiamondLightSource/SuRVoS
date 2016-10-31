---
title: Introduction to SuRVoS
layout: documentation
toc-section: 0
toc-title: Introduction
---

SuRVoS brings together machine learning models, computer vision techniques and human knowledge within a user interface to interactively segment large 3D volumes.

![]({{site.baseurl}}/images/survos_intro.png)

The user interface of SuRVoS is divided in two panels. On the left hand is the panel that contains all the available plugins (or tools). Each of the tools and its uses will be explained in the following sections. On the right panel, instead, resides the visualisation panel (where everything made with SuRVoS will be visualised) and another two main workflows. SuRVoS is able to work with very large datasets by means of its Region Of Interest (ROI) system and a HDF5 workspace. Everything loaded with SuRVoS and every action made within it will be saved in a workspace folder inside an HDF5 file. HDF5 is an on-disk data format that loads only the required data on memory. By using ROIs, SuRVoS limits all its actions to the selected ROI, which allows the user to load large datasets but operate only in small windows, also improving the performance of all the subsequent actions.

The following image summarizes the purpose and workflow of SuRVoS:

![]({{site.baseurl}}/images/survos_summary.png)

Dark blue boundaries in the volume represent *supervoxel* boundaries, a 1st order Super Region. Supervoxels are formed by grouping similar and adjacent voxels together, making them a compact representation of the volume. supervoxels are applied in 3D (each supervoxel is more or less `10 x 10 x 10` voxels in the above image) and have a very good boundary adherence, which allows them to preserve strong volume boundaries. By introducing Super Regions (supervoxels and megavoxels) manual segmentation of structures is reduced to annotating a few regions without having to delineate boundaries. Just by selecting a few supervoxels whole organelles can be annotated without having to delineate them.

Another important part of SuRVoS is the model training. By using super regions and user annotations (elongated lines with no opacity in the above image), large volume areas can be automatically segmented by using machine learning algorithms that *learn* to expand the segmentation to the whole volume. To that end, SuRVoS has available a wide variety of features that can be tuned for each volume and used for training. Once a model is trained, it predicts the labels for the rest of the volume and refines the segmentation using spatial information as shown in the above figure.

Every action made in the tool can be visualised in order to guide the user through the segmentation process and further tuning the parameters to obtain better results. As an example, once a feature has been extracted (e.g. Laplacian of Gaussian) it can be visualised to inspect its effect and assess whether it is useful or not for the current volume.

![]({{site.baseurl}}/images/survos_features.png)
