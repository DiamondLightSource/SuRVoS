<!-- Copy and paste the converted output. -->


<h2>SuRVoS 2 Roadmap</h2>


<h4>Make Improved Interactive Segmentation Tool</h4>


SuRVoS has a powerful interactive segmentation workflow that uses scribbles rather than full masks. Scribbles work by using a single mark within a superpixel to assign it a label. Using scribbles allows a significantly reduced number of marks to be used to form masks for gold standard labeling. The ability to interactively inspect and modify the image processing pipeline used for superregion generation and prediction is also a key feature. Given that prediction is fast, SuRVoS allows an expert user to rapidly iterate and improve a segmentation. We want to keep and extend this workflow to larger image volumes and to enable CNN-based segmentation workflows.

<h4>Make efficient annotation and model training workflows for image analysis</h4>


Providing interactive tools that assist developing annotation using machine learning methods, such as using an object detector to develop annotation for a full segmentation of an image volume. This involves a GUI for  inspecting volumes and viewing and creating the annotation, integrated with a workflow for segmentation and/or detection as well as support for converting vector data to raster data (e.g. points or boxes to masks). An example workflow might consist of taking crowd sourced points that label locations in an image and cleaning and inspecting the annotation. The annotation could then be used to train an object detector.

<h4>Support segmentation of big image data</h4>


With ever larger imaging detectors, volumes to be segmented can now reach sizes of hundreds of gigabytes. This raises a number of issues relating to opening the datafiles, the need for a responsive experience when carrying out interactive segmentation tasks, dividing and recombining the data for parallel processing on GPUs and segmentation on the differing scales of information contained in such huge volumes. By both supporting a workflow for breaking the data up into manageable chunks for processing and aggregation, as well as supporting deep learning models that can be trained on small annotated volumes but successfully generalize to larger volumes.

<h2>Module Overview </h2>


The basic volumetric image analysis modules are:



1. _Interactive Segmentation_
    1. **Method:** SuRVoS Super-region segmentation with MRF refinement
    2. **Input:** Scribbles and Shapes (hand or auto gen) on Volumetric image
    3. **Output:** Region-based segmentation
    4. **Annotation:** Scribbles. Imported vector data.
2. _Deep learning-based segmentation_
    1. **Method:** Using bounding box/mask annotation generated by previous steps, train an segmentation model (e.g. U-net, Faster/Mask-RCNN)
    2. **Output:** Object segmentations 
    3. **Annotation:** Masks, generated masks.
3. _Object detection and patch classification_
    1. **Method:** 2d/3d CNN image classification.
    2. **Output:** Region/Object location classifications.
    3. **Annotation:** Point and bounding volume annotation. 
4. _Big image data segmentation_
    1. **Method:** U-net and Dask for segmentation on computing clusters.
    2. **Output:** Semantic segmentation of entire image with accurate object boundaries.
    3. **Annotation:** Masks
5. _Crowdsourcing/Zooniverse_
    1. **Purpose:**  Crowdsourcing (Zooniverse) Subject Set Preparation and Classification Wrangling and Cleaning
    2. **Description**: Python module and associated GUI for preparing of subject sets for Zooniverse and subsequent import and processing of the data.
    3. **Output:** Semantic segmentation of entire image with accurate object boundaries.
6. _Interactive clustering_
    1. **Purpose**:  Interactive Clustering of super-regions and segmentation model interpretability. 
    2. **Answers**: “Which regions are similar to one another? How do the regions of the volume vary?
    3. **Description**: Interface that allows a user to run a clustering algorithm and view the clusters interactively 
    4. **Output**: Clusters as labeled regions 

<h2>SuRVoS2 Deliverables</h2>


<h4>Planning and Design</h4>




*   Plan features for replicating S1 workflow (loading existing S1 Workspace)
*   Plan interactive interface allowing similar use to the scribble/segment/change param workflow in S2.

<h4>Features</h4>




*   View and edit annotations (using Napari widget)
*   Deep segmenatic segmentation and object detection module. 
*   Filtering library (from S1). Possibility of use of Pytorch methods for auto-parameter selection.
*   Interactive segmentation prediction (extension of SuRVoS 1 methods)
*   Big Image Data, server-side segmentation module
*   Crowdsourced annotation utilities. Subject Set Generator for Zooniverse.
*   Interactive clustering of super-regions

<h4>Testing on Real world Data</h4>




*   Test S2a on Science Scribbler datasets as well as materials science datasets.

<h4>Products</h4>




*   S2a library documentation
*   S2a Prototype with Win and Linux builds
*   Repo (docs, nb, libs, conda env) for VF Workflow using S2a (see Science Scribbler VF project plan)
*   Repo (docs, nb, libs, conda env) for Huntington's Workflow using S2a (see Science Scribbler project plan)

<h2>Software architecture</h2>


<h3>Key technologies and libraries</h3>


CUDA, Pytorch, HDF5, EMDB-SFF, Dask, Napari

<h2>Related Software and Libraries</h2>




*   Smart annotation tools
    *   Commercial
        *   [https://docs.supervise.ly/neural-networks/overview/overview/](https://docs.supervise.ly/neural-networks/overview/overview/)
    *   [https://github.com/jsbroks/coco-annotator](https://github.com/jsbroks/coco-annotator)
    *   Deep Extreme Cut  [https://github.com/scaelles/DEXTR-PyTorch](https://github.com/scaelles/DEXTR-PyTorch)
*   Volumetric segmentation
    *   [http://dev.theobjects.com/dragonfly_4_1_release/contents.html](http://dev.theobjects.com/dragonfly_4_1_release/contents.html)
    *   [https://www.ilastik.org/documentation/index.html](https://www.ilastik.org/documentation/index.html)
    *   [https://github.com/MIC-DKFZ/medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
    *   [https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunet](https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunet)
*   Big image data
    *   The open source software Paintera ([https://github.com/saalfeldlab/paintera](https://github.com/saalfeldlab/paintera)) is designed for visualisation of 3D volumes of arbitrary size and utilises MIP maps to achieve this.
    *   Napari - a “fast, interactive, multi-dimensional image viewer” written in Python. There are examples of how to use napari for viewing large data in conjunction with the Dask library, allowing lazy loading of images here [https://napari.org/tutorials/dask](https://napari.org/tutorials/dask)
    *   Various ImageJ plugins such as BigDataViewer or SciView
*   Object detection and segmentation methods
    *   [https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
    *   [https://github.com/mapbox/robosat](https://github.com/mapbox/robosat)
    *   [https://github.com/chenyuntc/simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)
*   Watershed (2d/3d)
    *   [https://github.com/CellProfiler/tutorials/tree/master/3d_monolayer](https://github.com/CellProfiler/tutorials/tree/master/3d_monolayer)
*   Geodesic segmentation (2d/3d) 
    *   Scikit-fmm
    *   [https://paperswithcode.com/paper/deepigeos-a-deep-interactive-geodesic](https://paperswithcode.com/paper/deepigeos-a-deep-interactive-geodesic)
*   Interactive clustering
    *   Kepler Mapper [https://github.com/scikit-tda/kepler-mapper](https://github.com/scikit-tda/kepler-mapper)
    *   Web TSNE and Umap Visualisation [https://github.com/YaleDHLab/pix-plot](https://github.com/YaleDHLab/pix-plot)
