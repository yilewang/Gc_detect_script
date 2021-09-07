#!/usr/bin/python

"""
The script is designed to practice the workflow of how to preprocess fMRI data via Nipype package. The ultimate goal is to create my own workflow script. All the information can be accessed from https://miykael.github.io/nipype_tutorial/notebooks/handson_preprocessing.html website. This py file is only for practice.

The structure of the preprocessing workflow is below:

    1. Gunzip (Nipype)
    2. Drop Dummy Scans (FSL)
    3. Slice Time Correction (SPM)
    4. Motion Correction (SPM)
    5. Artifact Detection
    6. Segmentation (SPM)
    7. Coregistration (FSL)
    8. Smoothing (FSL)
    9. Apply Binary Mask (FSL)
    10. Remove Linear Trends (Nipype)

This script is for preprocessing only.

"""



import os
from os.path import join as opj
import json
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
import nipype.interfaces.spm as spm
import nipype.interfaces.matlab as matlab
from nipype.algorithms.rapidart import ArtifactDetect
from nipype import Workflow, Node




