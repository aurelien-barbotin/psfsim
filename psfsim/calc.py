# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 17:13:19 2018

@author: Aurelien
"""
import numpy as np

from scipy.integrate import nquad
import matplotlib.pyplot as plt
import time
import os

from zernike.czernike import RZern
from slm.slm import SLM

from scipy.signal import convolve
from scipy.optimize import minimize_scalar
from scipy import stats

import pkg_resources

#Theoretical FWHM: 275nm
#Axial 0.89*lambda/(n-sqrt(n**2-NA**2))
#Axial 721 nm
rz = RZern(8)

def distance_map(image,xc=0,yc=0):
    u,v=image.shape
    #Meshgrid usses cartesian indexing, invert of normal matrix indexing
    y = np.linspace(-u/2,u/2,u)-yc
    x = np.linspace(-v/2,v/2,v)-xc
    xx,yy = np.meshgrid(x,y)
    return np.sqrt(xx**2+yy**2)

class Simulator():

    def __init__(self,wavelength=755,NA=1.4,optical_n=1.518,sigma=0.5,npts=200,
                 amplitude = 1, res=20,nx=49,ny=49,nz=49,
                 th_on=True,helix_on=False,r1=0.61,
                 double_helix = False, cartesian_sampling = False):
        self.wl = wavelength
        self.NA = NA
        self.optical_n = optical_n
        self.sigma = sigma  #gaussian illumination std
        self.npts = npts
        self.amplitude = amplitude
        self.res = res
        self.nx,self.ny,self.nz = nx,ny,nz
        self.double_helix = double_helix
        #Not used yet
        self.th_on=th_on
        self.helix_on = helix_on
        self.r1=r1
        
        self.k = self.optical_n*2*np.pi/self.wl
        self.alpha = np.arcsin(NA/optical_n)
        self.moon_on = False
        self.cartesian_sampling = cartesian_sampling
        
    def A(self,theta):
        rho = self.optical_n*np.sin(theta)/self.NA
        if self.sigma>1000:
            return np.ones_like(theta)
        a = np.exp(-rho**2/(2*self.sigma**2))
        return a
    
    def phi_high_na_defocus(self,theta,phi):
        defoc = np.cos(theta)/(1-self.NA/self.n)
        return defoc
        
    def phi_ab(self,theta,phi,k):
        """Returns the induced aberration function. 
        Paramters: 
            theta: float, angle
            phi: float, angle
            k: int, zernike mode index"""
        rho = self.optical_n*np.sin(theta)/self.NA #Conversion
        radial = rz.radial(k, rho)
        ang = rz.angular(k, phi)
    
        return radial*ang
    
    def phi_pos(self,x,y,z,theta,phi):
        #xy
        sth = np.sin(theta)
        cth = np.cos(theta)
        phase = self.k*(
                x*sth*np.cos(phi)+
                y*sth*np.sin(phi)+
                z*np.cos(theta)
                )

        phase = np.exp(1j*phase)*sth*np.sqrt(cth)
        return phase
    
    def phi_th(self,theta,phi):
        """Top hat phase mask in spherical coordinates
        """
        rho = self.optical_n*np.sin(theta)/self.NA
        return (rho>=self.r1).astype(np.float)*np.pi

    def lineprofile(self,n,res,axis=0,**kwargs):
        """Calculates intensity profile along a given axis. 
        Parameters:
            n (int): number of datapoints to compute
            res (float): the distance between points in nm
            axis (int): optional, axis along which data is to be calculated.
                0: x, 1:y, 2: z
            kwargs (dict): supp arguments, such as aberrations. See intensity
        Returns:
            ndarray: (x,y) datapoints of lineprofile
            """
        xvals = np.arange(0,res*n,res)-res*(n//2)
        out = np.zeros(n)
        
        axes = [0,0,0]
        axes[axis] = 1
        axes = np.asarray(axes)
        
        for i,x in enumerate(xvals):
            out[i]=self.intensity(*axes*x,**kwargs)
        return np.array([xvals,out])
    
    def amplitude_2d(self):
        npts = self.npts
        thetas = np.linspace(0,self.alpha,npts)
        phis = np.linspace(0,2*np.pi,npts)
        ths,phs = np.meshgrid(thetas,phis)
        return self.A(ths)
    
    def cart2polar(self,xx,yy):
        """Transforms cartesian coordinates x,y generate from meshgrid into
        polar coordinates theta, phi"""
        rho = np.sqrt(xx**2+yy**2)
        phs = np.arctan2(xx,yy)
        #phs = phs[rho<=1]
        #rho = rho[rho<=1]
        ths = np.arcsin(rho*np.sin(self.alpha))
        # dphi = 1/npts
        # dtheta = dphi
        assert(ths.size==phs.size)
        return ths,phs
    
    def cart(self):
        """Generates cartesian coordinates"""
        npts = self.npts
        xx = np.linspace(-1,1,npts)
        yy = np.linspace(-1,1,npts)
        xx,yy = np.meshgrid(xx,yy)
        return xx,yy
    
    def polar(self):
        """Generates polar coordinates"""
        npts = self.npts
        thetas = np.linspace(0,self.alpha,npts)
        phis = np.linspace(0,2*np.pi,npts)
        ths,phs = np.meshgrid(thetas,phis)
        return ths, phs
        
    def intensity(self,x,y,z,aberration = None):
        npts = self.npts
        if self.cartesian_sampling:
            # !!! Needs to be updated?
            xx = np.linspace(-1,1,npts)
            yy = np.linspace(-1,1,npts)
            xx,yy = np.meshgrid(xx,yy)
            rho = np.sqrt(xx**2+yy**2)
            phs = np.arctan2(xx,yy)
            phs = phs[rho<=1]
            rho = rho[rho<=1]
            na = self.NA
            optical_n = self.optical_n
            ths = np.arctan(na/optical_n*rho/np.sqrt(1-(na/optical_n)**2))
            dphi = 1/npts
            dtheta = dphi
            assert(ths.size==phs.size)
            # thetas = np.linspace(0,self.alpha,npts)
            # phis = np.linspace(0,2*np.pi,npts)
        else:
            thetas = np.linspace(0,self.alpha,npts)
            phis = np.linspace(0,2*np.pi,npts)
            ths,phs = np.meshgrid(thetas,phis)
        
            dphi = (phis[-1]-phis[0])/npts
            dtheta = (thetas[-1]-thetas[0])/npts
        
            ths = ths.reshape(-1)
            phs = phs.reshape(-1)
            
        if self.th_on:
            phase = self.phi_pos(x,y,z,ths,phs) * np.exp(1j* self.phi_th(ths,phs))
        else:
            phase = self.phi_pos(x,y,z,ths,phs)
        if self.helix_on:
            factor = 1
            if self.double_helix:
                factor = 2
            phase*=np.exp(1j*factor * phi_helix(ths,phs))
        if self.moon_on:
            phase*=np.exp(1j*phi_moon(ths,phs))
        #Take into account amplitude variations
        phase*=self.A(ths)
        polarisation = e_eq2(ths,phs)
        
        if aberration is not None:
            phase_aberration=0
            for k,bias in enumerate(aberration):
                if bias!=0:
                    phase_aberration+=self.phi_ab(ths,phs,k)*bias
            phase*=np.exp(1j*phase_aberration)
            
        if not self.cartesian_sampling:
            assert polarisation.shape==(3,npts**2)
            assert phase.shape==(npts**2,)

        out = np.dot(polarisation,phase)*dtheta*dphi
        real = np.real(out)
        imag = np.imag(out)
        out = real**2+imag**2
        out = self.amplitude*out
        assert out.size==3
        return np.sum(out)    

    def integrate_mask(self,x,y,z,ampl_mask,phase_mask,aberration = None):
        npts = self.npts
        if self.cartesian_sampling:
            # !!! Needs to be updated?
            xx = np.linspace(-1,1,npts)
            yy = np.linspace(-1,1,npts)
            xx,yy = np.meshgrid(xx,yy)
            rho = np.sqrt(xx**2+yy**2)
            rhoc = rho.copy()
            phs = np.arctan2(xx,yy)
            phs = phs[rho<=1]
            rho = rho[rho<=1]
            ths = np.arcsin(rho*np.sin(self.alpha))
            dphi = 1
            dtheta = 1
            assert(ths.size==phs.size)
            phase_mask = phase_mask[rhoc<=1].reshape(-1)
            ampl_mask = ampl_mask[rhoc<=1].reshape(-1)
            # thetas = np.linspace(0,self.alpha,npts)
            # phis = np.linspace(0,2*np.pi,npts)
        else:
            thetas = np.linspace(0,self.alpha,npts)
            phis = np.linspace(0,2*np.pi,npts)
            ths,phs = np.meshgrid(thetas,phis)
        
            dphi = (phis[-1]-phis[0])/npts
            dtheta = (thetas[-1]-thetas[0])/npts
        
            ths = ths.reshape(-1)
            phs = phs.reshape(-1)
            
        
        phase = self.phi_pos(x,y,z,ths,phs) * phase_mask
        
        if self.moon_on:
            phase*=np.exp(1j*phi_moon(ths,phs))
        if self.th_on:
            phase*=np.exp(1j* self.phi_th(ths,phs,r1=self.r1))
        if self.helix_on:
            phase*=np.exp(1j * phi_helix(ths,phs))
        #Take into account amplitude variations
        phase*=self.A(ths)
        polarisation = e_eq2(ths,phs)
        
        if aberration is not None:
            phase_aberration=0
            for k,bias in enumerate(aberration):
                if bias!=0:
                    phase_aberration+=self.phi_ab(ths,phs,k)*bias
            phase*=np.exp(1j*phase_aberration)
            
        if not self.cartesian_sampling:
            assert polarisation.shape==(3,npts**2)
            assert phase.shape==(npts**2,)

        out = np.dot(polarisation,phase)*dtheta*dphi
        real = np.real(out)
        imag = np.imag(out)
        out = real**2+imag**2
        out = self.amplitude*out
        assert out.size==3
        return np.sum(out)
    
    def xyprof(self,n,res,z=0,**kwargs):
        xvals= np.arange(0,res*n,res)-res*(n//2)
        yvals= np.arange(0,res*n,res)-res*(n//2)
        out = np.zeros((n,n))
    
        for i,x in enumerate(xvals):
            print("line ",i,"out of",n)
            for j,y in enumerate(yvals):
                out[i,j]=self.intensity(x,y,z,**kwargs)
        self.current_image=out
        return out

    def xzprof(self,nx,nz,res,**kwargs):

        xvals= np.arange(0,res*nx,res)-res*(nx//2)
        zvals= np.arange(0,res*nz,res)-res*(nz//2)
        out = np.zeros((nz,nx))
        for i,x in enumerate(xvals):
            print("line ",i,"out of",nx)
            for j,z in enumerate(zvals):
                out[j,i]=self.intensity(x,0,z,**kwargs)
        return out
    
    def volume(self,nx,ny,nz,res,aberration=None):

        xvals= np.arange(0,res*nx,res)-res*(nx//2)
        yvals= np.arange(0,res*ny,res)-res*(ny//2)
        zvals= np.arange(0,res*nz,res)-res*(nz//2)
        out = np.zeros((nz,nx,ny))
        t0 = time.time()
        for i in range(xvals.size):
            x = xvals[i]
            print("line ",i,"out of",nx)
            t1 = time.time()
            print("Elapsed time betweem prev iteration and now:",t1-t0)
            t0 = t1
            for l,y in enumerate(yvals):
                for j in range(zvals.size):
                    z = zvals[j]
                    out[j,i,l]=self.intensity(x,y,z,aberration=aberration)
        return out
    
    def compute_disp_free(self,mode1=1,mode2=7,xy=True,r1=0.59,maxab = 0.3):
        minab = 0
        abrange=np.linspace(minab,maxab,5)
        
        mode1_res=[]
        mode2_res=[]
        print("psize",self.res)
        psize=self.res
        if xy:
            npx=self.nx
        else:
            npx = self.nz
        x= np.arange(0,self.res*npx,self.res)-self.res*(npx//2)
        
        def find_min(data,x):
            return x[np.where(data==np.min(data))[0][0]]
        sigma = find_sigma_for_r(r1)
        self.sigma = sigma
        
        for ab in abrange:
            
            aberration = np.zeros(15)
            aberration[mode1]=ab
            if xy:
                if mode1==1:
                    axis=0
                else:
                    axis=1
                depletion_1,xvals = self.lineprofile(npx,psize,aberration=aberration,axis=axis)
            else:
                depletion_1,xvals = self.lineprofile(npx,psize,aberration=aberration,axis=axis)

            aberration = np.zeros(15)
            aberration[mode2]=ab
            if xy:
                if mode1==1:
                    axis=0
                else:
                    axis=1
                depletion_2,xvals = self.lineprofile(npx,psize,
                                                     aberration=aberration,axis=axis)
            else:
                depletion_2,xvals = self.lineprofile(npx,psize,
                                                     aberration=aberration)
                
            depletion_1 = depletion_1.reshape(-1)
            depletion_2= depletion_2.reshape(-1)
            assert(np.all(x==xvals))
            plt.figure()
            plt.plot(x,depletion_1,x,depletion_2)
            plt.title(str(ab))
            
            mode1_res.append(find_min(depletion_1,x))
            mode2_res.append(find_min(depletion_2,x))
            
            plt.axvline(mode1_res[-1])
            plt.axvline(mode2_res[-1])
            plt.legend([mode1, mode2])
        slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(abrange,np.asarray(mode1_res))
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(abrange,np.asarray(mode2_res))
        
        print("ratio",mode2,"/",mode1,slope2/slope1)
        print("ratio2:",slope2)
        print("slope 1",slope1,"nm/rad")
        print("slope 2",slope2,"nm/rad")
        plt.figure()
        plt.plot(abrange,mode1_res,abrange,abrange*slope1+intercept1)
        plt.plot(abrange,mode2_res,abrange,abrange*slope2+intercept2)
        plt.title("Fitting displacements")
        plt.legend(["Mode 1","Mode 1 fit","Mode 2","Mode 2 fit"])
        return mode1_res,mode2_res
    
    def find_displacement(self,mode,npts=4,maxab=0.5,axis=0,abrange=None):
        if abrange is None:
            abrange=np.linspace(0,maxab,npts)
        
        mode1_res=[]
        psize=self.res
        npx = (self.nx,self.ny,self.nz)[axis]

        def find_min(data,x):
            
            if self.th_on or self.helix_on:
                return x[np.where(data==np.min(data))[0][0]]
            else:
                return x[np.where(data==np.max(data))[0][0]]
        
        """sigma = find_sigma_for_r(r1)
        self.sigma = sigma"""
        for ab in abrange:
            if mode<15:
                aberration = np.zeros(15)
            else:
                aberration = np.zeros(45)
            aberration[mode]=ab

            depletion_1,xv = self.lineprofile(npx,psize,axis=axis,aberration=aberration)

            plt.figure()
            plt.plot(xv,depletion_1)
            plt.title(str(ab))
            
            mode1_res.append(find_min(depletion_1,xv))
            
            plt.axvline(mode1_res[-1])
            
        slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(abrange,np.asarray(mode1_res))
        plt.figure()
        plt.plot(abrange,np.asarray(mode1_res),"bo")
        plt.plot(abrange,slope1*abrange+intercept1)
        plt.legend(["Calculated value","fit"])
        plt.xlabel("Aberration (rad)")
        plt.ylabel("displacement (nm)")
        plt.title("Mode "+str(mode))
        print("Mode:",mode,"Motion induced:",slope1,"nm/rad")
        return mode1_res,slope1
   
def phi_helix(theta,phi):
    return phi

def phi_moon(theta,phi,a=0):
    return (np.mod(phi,2*np.pi)>=np.pi).astype(np.float)*np.pi

def e_eq2(theta,phi):
    """Contains also the last sin theta term"""
    cth = np.cos(theta)
    sth = np.sin(theta)
    try:
        u = theta.size
        out = np.zeros((3,u),dtype=np.complex)
    except:
        u=0
        out = np.zeros(3,dtype=np.complex)
    out[0] = cth+1+(cth-1)*np.exp(1j*2*phi)
    out[1] = 1j*(cth+1)-1j*(cth-1)*np.exp(1j*2*phi)
    out[2] = -2*sth*np.exp(1j*phi)
    return out
     
def find_sigma_for_r(r1):
    """Returns the standard deviation of Gaussian illumination required to have 
    a specific z-STED inner radius"""
    
    file = pkg_resources.resource_filename('psfsim', 'data/r1_50pts.npy')
    print(file)
    out = np.load(os.path.join(file,file))
    sigmas = out[:,0]
    r1s=out[:,1]
    
    diffs = np.abs(r1s-r1)
    w1 = np.where(diffs==np.min(diffs))[0][0]
    if diffs[w1+1]>diffs[w1-1]:
        w2=w1-1
    else:
        w2=w1+1
        
    dist = np.abs(diffs[w1]-diffs[w2])
    d1=np.abs(r1-r1s[w1])
    d2=np.abs(r1-r1s[w2])
    
    v1,v2 = r1s[w1],r1s[w2]
    v =(v1 *(dist-d1) +v2*(dist-d2) )/dist
    v=sigmas[w1]
    return v

def find_min(data,x):
    return x[np.where(data==np.min(data))[0][0]]

def compute_mask_radius(sigma,minbound=0.4,maxbound=0.8):
    sim = Simulator(sigma=sigma,npts=800,amplitude=11441)
    f = lambda x :sim.intensity(0,0,0,top_hat_on=True,r1=x)
    minimized = minimize_scalar(f,method="bounded",bounds=[minbound,maxbound])
    return minimized

def compute_sigma_r1_correspondances(sigma0,sigma1,npts):
    sigmas = np.linspace(sigma0,sigma1,npts)
    r1s=np.zeros_like(sigmas)
    for i,sig in enumerate(sigmas):
        print("Progressing ",(i*100.0/sigmas.size),"%")
        r1s[i] = compute_mask_radius(sig,minbound=0.5,maxbound=0.75).x
    return sigmas,r1s

def FWHM(line):
    mm = np.where(line==np.max(line))[0][0]
    linel=line[:mm]
    liner=line[mm:]
    def find_half(semiline):
        diffs = np.abs(semiline-np.max(line)/2)
        return np.where(diffs==np.min(diffs))[0][0]
    h1 = find_half(linel)
    h2 = find_half(liner)+mm
    fwhm = h2-h1
    return fwhm,h1,h2

    
def attenuation(exc,depl,P,delta=0):
    """STED attenuation formula. 
    Parameters:
        exc: numpy array, excitation
        depl: numpy array, depletion
        P: float, STED power in fractions of Psat
        delta: optional, unmodulated fraction"""
    assert(delta<1)
    #Consider here Isat=1
    #return exc * 1/np.sqrt(1+depl*P)
    return (1-delta)*exc*np.exp(-depl*P)+delta
#test_slm_mode_integral()
def poisson_noise(image,counts=None):
    out = image.copy()
    if counts is not None:
        out/=np.max(out)
        out*=counts
    out = np.random.poisson(out)
    return out

def pinhole(radius,res,stack):
    nz  =stack.shape[0]
    pinhole = (distance_map(stack[0])<=radius/res).astype(np.float)
    out = np.zeros_like(stack)
    for i in range(nz):
        out[i]=convolve(stack[i],pinhole,mode="same")
    out /= np.max(out)
    return out

def pinhole1D(radius,res,stack):
    nx  = stack.shape[1]
    nz = stack.shape[0]
    pinhole = np.abs(np.linspace(-nx/2,nx/2,nx)*res)
    pinhole = (pinhole<radius).astype(np.float)
    out = np.zeros_like(stack)
    for i in range(nz):
        out[i]=convolve(stack[i],pinhole,mode="same")
    out /= np.max(out)
    return out

if __name__=='__main__':
    #plt.close("all")
    
    sigma = find_sigma_for_r(0.61)
    depletion = Simulator(sigma=sigma,npts=100,wavelength = 755,double_helix=False)
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
    #out = depletion.compute_disp_free(maxab=0.5,mode1=1,mode2=7)
    
    #out = depletion.find_displacement(21,npts=8,axis=2)
    #aberration[6] = 0.2
    #out=depletion.volume(50,50,50,40,aberration = aberration)
    #show_3Dprofiles(out)
    1/0
    
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
    

    
