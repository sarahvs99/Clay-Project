#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 12:02:04 2021

@author: sarah
"""

import MDAnalysis as mda
from MDAnalysis.analysis import lineardensity as lin
from MDAnalysis.analysis import density
from MDAnalysis.topology import tpr
from MDAnalysis.topology import TPRParser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_universe():
    