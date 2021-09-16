#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:27:21 2020

@author: aurelien
"""
import matplotlib.pyplot as plt
import numpy as np

from psfsim.calc import find_sigma_for_r, Simulator

def example1():
    """Simulates the axial intensity profile of a 2D-STED depletion beam"""
    sigma = find_sigma_for_r(0.61)
    depletion = Simulator(sigma=sigma,npts=100,wavelength = 755)
    depletion.nx=100
    depletion.ny=100
    depletion.nz = 200
    depletion.res = 5
    depletion.r1 = 0.61
    depletion.sigma = sigma
    
    depletion.th_on = 0
    depletion.helix_on = 1
    depletion.moon_on = 0
    
    xz1 = depletion.xzprof(50,100,40)
    plt.figure()
    plt.imshow(xz1)
    plt.show()
    return xz1

def example2():
    """Calculates the xz excitation PSF of a confocal microscope"""
    sigma = find_sigma_for_r(0.61)
    excitation = Simulator(sigma=sigma,npts=100,wavelength = 640)
    excitation.nx=100
    excitation.ny=100
    excitation.nz = 200
    excitation.r1 = 0.61
    excitation.th_on = 0
    excitation.sigma = sigma
    xz2 =  excitation.xzprof(50,80,40)
    plt.figure()
    plt.imshow(xz2)
    plt.show()
    return xz2

def example3():
    """Simulates the axial intensity profile of an aberrated z-STED depletion beam"""
    sigma = find_sigma_for_r(0.61)
    depletion = Simulator(sigma=sigma,npts=100,wavelength = 755)
    depletion.nx=100
    depletion.ny=100
    depletion.nz = 200
    depletion.res = 5
    depletion.r1 = 0.61
    depletion.sigma = sigma
    
    depletion.th_on = 1
    depletion.helix_on = 0
    depletion.moon_on = 0
    aberration = np.zeros(15)
    aberration[10] = 0.7
    
    xz1 = depletion.xzprof(50,100,40, aberration = aberration)
    
    plt.figure()
    plt.imshow(xz1)
    plt.show()
    return xz1

xz = example3()
