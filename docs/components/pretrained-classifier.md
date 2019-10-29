---
title: Pre-trained classifier
layout: documentation
toc-section: 3
toc-subsection: 10
---



It is now possible to perform label prediction on data using a classifier that has been trained on a different dataset. This is likely to be particularly useful when segmenting a number of similar datasets, enabling the same model to applied across all of them, reducing the need for manual annotation.

## Saving a classifier

Training of, and subsequent prediction with, a machine learning classifier is performed by pressing the **Train & Predict** button which is found in the *Train Classifier* tab. After this step, the **Save Classifier** button at the bottom of the tab will become active. Pressing this button will bring up a dialog to save the classifier in a folder called *classifiers* which is generated automatically. The default filename is *classifier.h5*.

## The "Use Pre-trained Classifier" component

This feature is for use with new data loaded into a fresh workspace. First, start SuRVoS and then load in the appropriate dataset and create a workspace folder. 

![left30]({{site.baseurl}}/images/components/pretrained/pretrained_clf_component.png)

### 1. Load the classifier

In order to use a classifier that has been trained previously, the first step is to press the **Load Saved Classifier** button. A dialog box will appear for selection of the classifier file.

![right30]({{site.baseurl}}/images/components/pretrained/info_box.png)

After the classifier is loaded an information box will appear describing the type of classifier that has been loaded along with the label classes associated with that classifier.

The label names and associated colours will now be visible when switching to the *Annotations* tab. 

### 2. Calculate the required feature channels and supervoxels

In order to predict labels on new data using a saved classifier, the same feature channels need to be present as when the classifier was trained. In addition, in order to apply predictions to the data volume, supervoxels need to be calculated for the data.

![left30]({{site.baseurl}}/images/components/pretrained/info_box2.png)

The feature channel and supervoxel information is saved along with the classifier. In order to apply these saved settings, click the **Calculate Channels and Supervoxels** button. A confirmation dialog box will appear with information describing the feature filters to be applied to the data and the parameters for the supervoxels to be calculated. After clicking **Yes**, the required components will be calculated. Once calculated, the feature channels will be automatically selected in the **Select Sources** dropdown menu and the **Predict** button will become available to use.

### 3. Perform the prediction

To perform prediction, click the **Predict** button. If required, prediction refinement can be selected also, before performing the prediction step.

### (Optional) 4. Add additional training data, retrain and save a modified classifier

If the prediction is not as expected and needs modification, it is possible to go back to the *Annotations* tab, select a label and add annotations to the data to correct areas that have been misclassified. Upon returning to the *Pretrained classifier* tab, if the **Predict** button is pressed a dialog box will now appear asking if you would like to append this new data to the training data and train a new classifier. Upon choosing 'Yes', training and prediction of a new classifier will be performed and the **Save new classifier** button at the bottom of the tab will become activated. Pressing this button will bring up a dialog to save this modified classifier in a new file.

