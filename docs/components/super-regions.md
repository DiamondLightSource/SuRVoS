---
title: Super Regions
layout: documentation
toc-section: 3
toc-subsection: 4
---

![left30]({{site.baseurl}}/images/components/super-regions/sregions.png)

The volumetric data is represented as super regions within a 3-layer hierarchical structure. This structure is composed of voxels, supervoxels and megavoxels. Each of these layers is formed by grouping similar and nearby elements of the previous layer. That is, while voxels represent standard volume voxels, supervoxels are groups of adjacent voxels grouped together into a meaningful region that preserves strong volume boundaries (ie. boundaries between different biological features in the image). Similarly, megavoxels are groups of nearby supervoxels that have similar appearance. With this hierarchical partitioning, large areas of the volume belonging to the same object are represented by: a set of thousands of voxels, tens of supervoxels or a few megavoxels (Figure 4). This hierarchical structure has several advantages compared to the standard voxel grid:

1. Each of the layers of the hierarchy represents the same volume with many fewer elements than the previous layer, thus, reducing the complexity of annotating or segmenting the volume by several orders of magnitude.

2. Supervoxels and megavoxels are designed to have a strong boundary adherence and are therefore able to represent the same biological feature without significant loss of information.

3. The shape and size of the supervoxels is user-definable in order to properly model volumes or areas with different physical characteristics.

To better understand what super-regions actually do, find bellow a real world image of the [BSD500 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) that is easier to visually appreciate the differences.

![]({{site.baseurl}}/images/components/super-regions/abstract.png)

Here, a 2D image (left) is segmented into superpixels that preserver strong image boundaries (middle). These superpixels are then merged into *supersegments* (2D megavoxels, right) to further represent the image with less regions. This process can be equivalently applied to 3D images to represent it as supervoxels and megavoxels, reducing the amount of representative units (for a 1000x1000x500 image) from 500M voxels to 100-500K supervoxels, and further down to 10-50K megavoxels.

![]({{site.baseurl}}/images/components/super-regions/super-regions.png)

## Supervoxels

TODO: parameters

## Megavoxels

TODO: parameters
