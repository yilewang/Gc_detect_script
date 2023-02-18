#!/usr/bin/python

import os
import sys
from os.path import join as opj
import json
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
import nipype.interfaces.spm as spm
import nipype.interfaces.matlab as matlab
from nipype.algorithms.rapidart import ArtifactDetect
from nipype import Workflow, Node
import nipype.interfaces.afni as afni

"""
The main purpose of this workflow is to organize the data structure of the HSAM data set and convert all the NIFIT or AFNI format imaging data to qualified way for preprocessing by TVB-UKBB pipeline

"""

subj_path = str(sys.argv[1])

