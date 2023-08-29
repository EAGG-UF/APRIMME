#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:38:40 2023

@author: joseph.melville
"""


import functions as fs
import matplotlib.pyplot as plt
import numpy as np
import torch 
from tqdm import tqdm
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



