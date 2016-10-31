---
title: Feature Channels
layout: documentation
toc-section: 3
toc-subsection: 3
---

![left30]({{site.baseurl}}/images/components/features/feature-panel.png)

Features help to better visualize and describe the data. There are multiple features available in SuRVoS. Standard features include Gaussian denoising, local statistics (e.g. mean, std), local standarization, Gaussian gradient magnitude and orientation and some other filters. More advanced blob detection features such as Difference of Gaussians, Laplacian of Gaussians or Determinant of Hessian help to detect blobs (bright or dark) in the image and textural featues can be extracted from the Eigenvalues of the Hessian or Structure Tensor matrices. These filters can also be computed at multiple scales for scale invariance.

Every feature computed in SuRVoS can be visually inspected in the main panel by choosing an already computed feature in the *channel-selection* combo box (image on the right). This helps an unexperienced user to understand what features are actually enhancing in the image and whether they are useful or not. Default parameters for each of the features are sensibly chosed for general volumes, however, by using the feature visualization it helps to understand and find appropiate feature parameters for the volume being analyzed.

![right40]({{site.baseurl}}/images/components/features/features-demo.png)

To compute a new feature, it has to first be selected from the available feature list and then click in **Add**. After the feature is added to the list, it will show (by clicking on the name) the list of configurable parameters. Once the feature parameters are selected, the channel can be computed by clicking on the tick symbol next to the name. Alternatively, multiple features can be computed at once by clicking on **Compute Features**.

Additionally, every feature has a parameter that defines the *source channel* (combo box marked as red on the left image). This source indicates which channel will be used to compute the new feature. The default channel is the raw data, but, it allows to compute **stacked features** (features of features) with limitless level of customization. This allows experienced users to compute complex features that would better enhance elements of the data.

Bellow will be listed a brief description of each of the features and the effect of the parameters on them.

## Raw features

### Threshold

### Invert Threshold

## Denoising Filters

### Gaussian Filter

### Total Variation denoising

## Local Features

## Gaussian Features

## Blob Detection

### Difference of Gaussians

### Laplacian of Gaussians

### Determinant of Hessian

## Texture/Structure

### Eigenvalues of the Hessian Matrix

### Eigenvalues of the Structure Tensor

### Gabor-like filters

## Robust Features

### Scale Invariant

### Frangi Filter

## Actiation Layers

### Maximum Response

### Rectified Linear Unit
