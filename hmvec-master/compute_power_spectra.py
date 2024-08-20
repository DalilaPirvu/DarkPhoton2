import hmvec as hm
import numpy as np
import scipy as scp
from scipy.special import eval_legendre, legendre, spherical_jn
import itertools
import wigner
from sympy.physics.wigner import wigner_3j
import time
from scipy import interpolate
from itertools import cycle
from math import atan2,degrees,lgamma 
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp2d,interp1d
import scipy.interpolate as si

from scipy.optimize import fsolve
from scipy.interpolate import RegularGridInterpolator
import random
import seaborn as sns

from params import *
#from plotting import *

############### COMPUTE ANGULAR POWER SPECTRA ###########################

def get_fourier_to_multipole_Pkz(zs, ks, chis, ellMax, Pklin):
    ells = np.arange(ellMax)
    Pzell = np.zeros((len(zs), len(ells)))

    f = interp2d(ks, zs, Pklin, bounds_error=True)     
    for ii, ell in enumerate(ells):
        kevals = (ell+0.5)/chis
        interpolated = si.dfitpack.bispeu(f.tck[0], f.tck[1], f.tck[2], f.tck[3], f.tck[4], kevals, zs)[0]
        Pzell[:, ii] = interpolated
    return Pzell

# Compute crossing radius of the Milky Way
def dark_photon_conv_prob_MilkyWay(mMWvir, rMWvir, rhocritzMW, deltavMW, csMW, HMW, rE, rs, mDP, pick_prof, name='battagliaAGN'):
    if pick_prof:
        return dark_photon_conv_prob_MilkyWay_gas(mMWvir, rMWvir, rhocritzMW, deltavMW, csMW, HMW, rE, rs, mDP, name)
    else:
        return dark_photon_conv_prob_MilkyWay_NFW(mMWvir, rMWvir, csMW, HMW, rE, rs, mDP)

def dark_photon_conv_prob_MilkyWay_gas(mMWvir, rMWvir, rhocritzMW, deltavMW, csMW, HMW, rE, rs, mDP, name='battagliaAGN'):
    delta_rhos1 = deltavMW*rhocritzMW
    delta_rhos2 = 200.*rhocritzMW
    m200critzMW = hm.mdelta_from_mdelta_unvectorized(mMWvir, csMW, delta_rhos1, delta_rhos2)
    r200critzMW = hm.R_from_M(m200critzMW, rhocritzMW, delta=200.)
    gas_profile = get_gas_profile(rs, 0., m200critzMW, r200critzMW, rhocritzMW, name=name)

    if mDP**2. < np.max(conv*gas_profile):
        idx    = np.argmin(np.abs(conv*gas_profile - mDP**2.))
        rcross = rs[idx]
        dmdr   = np.abs(conv*get_deriv_gas_profile(rcross, 0., m200critzMW, r200critzMW, rhocritzMW, name=name))
        limits = 1.#np.heaviside(rMWvir-rcross, 1)#*np.heaviside(rcross-rE, 1.)
        prob   = np.pi*mpcEVinv*(mDP**4.)*limits/dmdr
    else:
        rcross, prob = np.nan, 0.
    return rcross, prob

def dark_photon_conv_prob_MilkyWay_NFW(mMWvir, rMWvir, csMW, HMW, rE, rs, mDP):
    rss = rMWvir/csMW
    zMW = 0.

    nfw_rhoscsales = hm.rhoscale_nfw(mMWvir, rMWvir, csMW)
    nfw_profiles   = hm.rho_nfw(rs, nfw_rhoscsales, rss)

    idx  = np.argmin(np.abs(conv*nfw_profiles - mDP**2.))
    rcross = rs[idx]

    rfr  = rcross/rss
    dmdr = np.abs(conv*(nfw_rhoscsales/rss)*(1.+3.*rfr)/(rfr)**2./(1.+rfr)**3.)
    limits = 1.#np.heaviside(rMWvir-rcross, 1)#*np.heaviside(rcross-rE, 1.)

    prob = np.pi*mpcEVinv*(mDP**4.)*limits/(1.+zMW)/dmdr
    return rcross, prob

def get_volume_conv(chis, Hz):
    # Volume of redshift bin divided by Hubble volume
    # Chain rule factor when converting from integral over chi to integral over z
    return chis**2. / Hz

def get_rcross_per_halo(zs, ms, rs, rvir, rhocritz, deltav, cs, mDP, pick_prof, name='battagliaAGN'):
    if pick_prof:
        return get_rcross_per_halo_gas(zs, ms, rs, rvir, rhocritz, deltav, cs, mDP, name)
    else:
        return get_rcross_per_halo_NFW(zs, ms, rs, rvir, cs, mDP)

def get_rcross_per_halo_NFW(zs, ms, rs, rvir, cs, mDP):
    """ Compute crossing radius of each halo
    i.e. radius where plasma mass^2 = dark photon mass^2
    Find the index of the radius array where plasmon mass^2 = dark photon mass^2 """

    rss    = rvir/cs
    rhos   = hm.rhoscale_nfw(ms[None,:], rvir, cs)

    rcross_res = np.zeros((len(zs), len(ms)))
    for i_z, z in enumerate(zs):
        for i_m, m in enumerate(ms):
            func = lambda x: np.abs(hm.rho_nfw(x, rhos[i_z, i_m], rss[i_z, i_m]) * conv/mDP**2. - 1.)
            rcross_res[i_z, i_m] = fsolve(func, x0=rs[0])
    return rcross_res

def get_rcross_per_halo_gas(zs, ms, rs, rvir, rhocritz, deltav, cs, mDP, name='battagliaAGN'):
    """ Compute crossing radius of each halo
    i.e. radius where plasma mass^2 = dark photon mass^2
    Find the index of the radius array where plasmon mass^2 = dark photon mass^2 """

    m200critz, r200critz = get_200critz(zs, ms, cs, rhocritz, deltav)

    rcross_res = np.zeros((len(zs), len(ms)))
    for i_z, z in enumerate(zs):
        for i_m, m in enumerate(ms):
            func = lambda x: np.abs(get_gas_profile(x, z, m200critz[i_z, i_m], r200critz[i_z, i_m], rhocritz[i_z], name=name) * conv/mDP**2. - 1.)
            rcross_res[i_z, i_m] = fsolve(func, x0=rs[0])
    return rcross_res


def get_200critz(zs, ms, cs, rhocritz, deltav):
    delta_rhos1 = rhocritz*deltav
    delta_rhos2 = 200.*rhocritz
    m200critz = hm.mdelta_from_mdelta(ms, cs, delta_rhos1, delta_rhos2)
    r200critz = hm.R_from_M(m200critz, rhocritz[:,None], delta=200.)
    return m200critz, r200critz

def get_gas_profile(rs, zs, m200, r200, rhocritz, name='battagliaAGN'):
    #choose profile
    if name=='ACT': rho0, alpha, beta, gamma, xc = bestfitACT()
    elif name=='battagliaSH': rho0, alpha, beta, gamma, xc = battagliaSH(m200, zs)
    else: rho0, alpha, beta, gamma, xc = battagliaAGN(m200, zs)

    rho = rhocritz * rho0
    x = rs/r200/xc
    expo = -(beta+gamma)/alpha     # gamma sign must be opposite from Battaglia/ACT paper; typo
    return rho * (x**gamma) * ((1.+x**alpha)**expo)

def get_halo_skyprofile(zs, chis, rcross):
    # get bounds of each regime within halo
    rchis = chis*aa(zs)
    fract = (rcross/rchis[:,None])[None,...]

    listincr = 1. - np.geomspace(1e-3, 1., 41)
    listincr = np.asarray([1.] + listincr.tolist())[::-1]
    angs = listincr[:,None,None] * fract

    ucosth = (1.-(angs/fract)**2.)**(-0.5)
    ucosth[angs == fract] = 0.
    return ucosth, angs

def get_u00(zs, chis, rcross):
    # this gives the analytical result for the monopole
    rchis = chis*aa(zs)
    fract = (rcross/rchis[:,None])
    return fract**2./2.

def get_uell0(angs, ucosth, ell):
    # this returns the analytical approximation for low ell
    # or numerical result for higher multipoles

    uL0 = np.zeros(angs[0].shape)

    approx = ell < 0.1/angs[-1, :, :] # indices for which we can use the approximation

    uL0[approx] = 2.*np.pi * (angs[-1, :, :][approx])**2.

    # angular function u(theta) is projected into multipoles
    cos_angs = np.cos(angs[:, ~approx])
    Pell     = eval_legendre(ell, cos_angs)
    integr   = Pell * np.sin(angs[:, ~approx]) * ucosth[:, ~approx]
    uL0[~approx] = 2.*np.pi * np.trapz(integr, angs[:,~approx], axis=0)

    if ell%100==0:
        print(ell)
    return uL0 * ((4.*np.pi) / (2.*ell+1.))**(-0.5)

def get_deriv_gas_profile(rs, zs, m200, r200, rhocritz, name='battagliaAGN'):
    #choose profile
    if name=='ACT': rho0, alpha, beta, gamma, xc = bestfitACT()
    elif name=='battagliaSH': rho0, alpha, beta, gamma, xc = battagliaSH(m200, zs)
    else: rho0, alpha, beta, gamma, xc = battagliaAGN(m200, zs)

    x = rs / r200 / xc
    P = rhocritz * rho0
    expo = -(alpha+beta+gamma)/alpha
    drhodr = P * (x**gamma) * (1. + x**alpha)**expo * (gamma - x**alpha * beta) / rs
    
    if hasattr(rs, "__len__"): drhodr[rs==0.] = 0.
    elif rs==0: drhodr = 0.
    return drhodr

def get_thomsontau_per_halo(zs, ms, ellMax, chis, rvirs, rhocritz, deltav, cs, mDP, name='battagliaAGN'):
    m200, r200 = get_200critz(zs, ms, cs, rhocritz, deltav)
    rit1 = 1e-3 + 1. - np.geomspace(1., 1e-3, 20)
    rit2 = np.geomspace(1e-3, 1., 20)
    rits = np.sort(np.concatenate(([1e-10], rit1, rit2), axis=0))

    mult = 4.*np.pi * conv2 * aa(zs)/chis**2.
    mult = mult[None,:,None]
    print(np.shape(mult), mult)

    rs = rits[None,None,:] * rvirs[...,None]
    gas_profile = get_gas_profile(rs, zs[:,None,None], m200[...,None], r200[...,None], rhocritz[:,None,None], name='battagliaAGN')
    print(np.shape(gas_profile), gas_profile)

    ellsss = np.arange(ellMax) + 0.5

    rsfr = rs / chis[:,None,None]
    fact = ellsss[:, None,None,None] * rsfr[None, ...]
    factt= np.sin(fact) / fact
    print(np.shape(factt), factt)

    out = mult * np.trapz( (rs**2. * gas_profile)[None,...] * factt, rs, axis=-1)
    return out

def dark_photon_conv_prob(zs, ms, rs, rvir, rhocritz, deltav, cs, mDP, rcross, pick_prof, rscale=False, name='battagliaAGN'):
    if pick_prof:
        return dark_photon_conv_prob_gas(zs, ms, rs, rvir, rhocritz, deltav, cs, mDP, rcross, name)
    else:
        return dark_photon_conv_prob_NFW(zs, ms, rs, rvir, cs, mDP, rcross, rscale)

def dark_photon_conv_prob_gas(zs, ms, rs, rvir, rhocritz, deltav, cs, mDP, rcross, name='battagliaAGN'):
    m200, r200 = get_200critz(zs, ms, cs, rhocritz, deltav)
    drhodr = get_deriv_gas_profile(rcross, zs[:,None], m200, r200, rhocritz[:,None], name=name)
    dmdr   = np.abs(conv*drhodr)
    omgz   = (1.+zs[:,None])#*omega0 it doesn't change the phenomenology, but we want to remove frequency dependence later.
    uang   = 2.*np.heaviside(rvir - rcross, 0.5)
    return np.pi * mpcEVinv * (mDP**4.) * uang / omgz / dmdr

def dark_photon_conv_prob_NFW(zs, ms, rs, rvir, cs, mDP, rcross, rscale=False):
    rss  = rvir/cs
    rhos = hm.rhoscale_nfw(ms[None,:], rvir, cs)
    rfr  = rcross/rss
    dmdr = np.abs(conv*(rhos/rss)*(1.+3.*rfr)/(rfr)**2./(1.+rfr)**3.)
    omgz = (1.+zs[:,None])#*omega0 it doesn't change the phenomenology, but we want to remove frequency dependence later.

    if rscale: uang = 2.*np.heaviside(rvir-rcross, 0.5) * np.heaviside(rcross-rss, 1.)
    else: uang = 2.*np.heaviside(rvir-rcross, 0.5)
    return np.pi * mpcEVinv * (mDP**4.) * uang / omgz / dmdr

def get_avtau(zs, ms, nzm, dvol, prob00):
    # Average optical depth per redshift bin
    dtaudz = np.trapz(nzm * prob00, ms, axis=-1) * dvol * 4*np.pi
    avtau  = np.trapz(dtaudz, zs, axis=0)
    return avtau, dtaudz

def get_scrCLs(l0Max, CMB, DPCl):
    l0List   = np.arange(   l0Max)
    ell1_CMB = np.arange(2, l0Max)
    ell2_scr = np.arange(2, l0Max)

    TTCl, EECl, BBCl, TECl = CMB[:,0], CMB[:,1], CMB[:,2], CMB[:,3]

    every_pair = np.asarray(list(itertools.product(ell1_CMB, ell2_scr)))
    allcomb = len(every_pair)
    nums    = np.array(np.linspace(0, allcomb, 20), dtype=int).tolist()

    scrTT, scrEE, scrBB, scrTE = np.zeros(l0Max), np.zeros(l0Max), np.zeros(l0Max), np.zeros(l0Max)
    wig000 = np.zeros(l0Max)
    wig220 = np.zeros(l0Max)
    for ind, (l1,l2) in enumerate(every_pair):

        if ind in nums: print(ind, 'out of', allcomb, nums.index(ind)+1)

        norm = (2.*l1+1.)*(2.*l2+1.)/(4.*np.pi)

        w000   = wigner.wigner_3jj(l1, l2, 0, 0)
        am, bm = max(2, int(w000[0])), min(int(w000[1]), l0Max-1)
        l000   = np.arange(am, bm+1)
        aw, bw  = int(am - w000[0]), int(bm - w000[0])

        wig000[:] = 0.
        wig000[l000] = w000[2][aw:bw+1]

        w220   = wigner.wigner_3jj(l1, l2, 2, 0)
        cm, dm = max(2, int(w220[0])), min(int(w220[1]), l0Max-1)
        l220   = np.arange(cm, dm+1)
        cw, dw  = int(cm - w220[0]), int(dm - w220[0])

        wig220[:] = 0.
        wig220[l220] = w220[2][cw:dw+1]

        scrTT += norm * DPCl[l2] * TTCl[l1] * np.abs(wig000)**2.
        scrTE += norm * DPCl[l2] * TECl[l1] * wig000 * wig220

        mix    = norm * DPCl[l2] * EECl[l1] * np.abs(wig220)**2.
        Jell   = l0List+l1+l2
        delte  = 0.5*(1. + (-1.)**Jell)
        delto  = 0.5*(1. - (-1.)**Jell)

        scrEE += mix * delte
        scrBB += mix * delto
    return l0List, scrTT, scrEE, scrBB, scrTE


def noise(ellMax, nfreqs, experiment):
    # Instrumental noise: takes parameters Beam FWHM and Experiment sensitivity in T
    ''' Output format: (spectrum type, ells, channels)'''

    beamFWHM = experiment['FWHMrad']**2.
    beamFWHM = beamFWHM[:nfreqs]
    deltaT   = experiment['SensitivityμK']**2.
    deltaT   = deltaT[:nfreqs]
    lknee    = experiment['Knee ell']
    aknee    = experiment['Exponent']

    ells = np.arange(2, ellMax)
    rednoise = ((ells/lknee)**aknee if (lknee!=0. and aknee!=0.) else 0.)
    ellexpo  = ells * (ells + 1.) / (8. * np.log(2))

    NellTT = np.zeros((ellMax, nfreqs, nfreqs))
    Beams  = np.zeros((ellMax, nfreqs))
    for frq in range(nfreqs):
        NellTT[2:,frq,frq]= deltaT[frq] * ( 1. + rednoise )
        Beams[2:,frq] = np.exp(-ellexpo * beamFWHM[frq])

    Beams2D  = (Beams[:,None,:] * Beams[:,:,None])**0.5

    NoiseTT = NellTT / Beams2D
    NoiseTT[np.isnan(NoiseTT)] = 0.
    return np.array([NoiseTT, np.sqrt(2)*NoiseTT, np.sqrt(2)*NoiseTT, np.zeros(np.shape(NoiseTT))])


def get_ILC_noise(ellMax, units0, screening, foregs, recCMB, experiment, nspec=4, nfreqs=9):
    freqs  = experiment['freqseV'][:nfreqs]
    units  = xov(freqs) / freqs
    freqMAT= units0**2. / np.outer(units, units)
    ee     = np.ones(nfreqs)
    onesMAT= np.outer(ee, ee)

    weights = np.zeros((nspec, ellMax, nfreqs))
    leftover= np.zeros((nspec, ellMax))
    elltodo = np.arange(2, ellMax)
    for spec in range(nspec):
        for ell in elltodo:
            CellBBω2= freqMAT * recCMB[spec,ell]
            Cellττ  = onesMAT * screening[spec,ell]
            Fellω2  = freqMAT * foregs[spec,ell]
            try:
                Cellinv = scp.linalg.inv(CellBBω2 + Fellω2 + Cellττ)
            except:
                print(ell, spec, recCMB[spec,ell])
                print(screening[spec,ell])
                print(noiseinstr[spec,ell])
                continue

            weights[spec,ell] = (Cellinv@ee)/(ee@Cellinv@ee)
            leftover[spec,ell]= weights[spec,ell]@(CellBBω2 + Fellω2)@weights[spec,ell]
    return weights, leftover

def get_ILC_BB_noise(ellMax, screening, noiseinstr, recCMB, nspec=4, nfreqs=9):
    ee     = np.ones(nfreqs)
    onesMAT= np.outer(ee, ee)

    weights = np.zeros((nspec, ellMax, nfreqs))
    leftover= np.zeros((nspec, ellMax))
    elltodo = np.arange(2, ellMax)
    for spec in range(nspec):
        for ell in elltodo:
            CellBBω2= np.diag(ee) * recCMB[spec,ell]
            Cellττ  = onesMAT * screening[spec,ell]
            Nellω2  = noiseinstr[spec,ell]
            try:
                Cellinv = scp.linalg.inv(CellBBω2 + Nellω2 + Cellττ)
            except:
                print(ell, spec, CellBBω2[spec,ell])
                print(Cellττ[spec,ell])
                print(noiseinstr[spec,ell])
                continue

            weights[spec,ell] = (Cellinv@ee)/(ee@Cellinv@ee)
            leftover[spec,ell]= weights[spec,ell]@(CellBBω2 + Nellω2)@weights[spec,ell]
    return weights, leftover


def taureco_NEdscBdsc(l0Max, ClEErec, ClBBrec, NlEE, NlBB):
    l0List = np.arange(   l0Max)
    l1List = np.arange(2, l0Max)
    l2List = np.arange(2, l0Max)

    all_possible_pairs = np.asarray(list(itertools.product(l1List, l2List)))

    sumell = np.zeros(l0Max)
    for l1,l2 in all_possible_pairs:
        wig3j   = wigner.wigner_3jj(l1, l2, -2,  2)
        wig3jre = wigner.wigner_3jj(l1, l2,  2, -2)

        am, bm  = int(wig3j[0]), min(int(wig3j[1])+1, l0Max-1)
        lwig    = np.arange(am, bm)

        pad       = np.zeros(l0Max)
        pad[lwig] = wig3j[2][lwig-am] - wig3jre[2][lwig-am]

        norm    = (2.*l1+1.)*(2.*l2+1.)*(2.*l0List+1.)/(4.*np.pi)
        gaEB    = norm * np.abs(-0.5j * ClEErec[l1] * pad)**2.
        #denom   = (ClEErec[l1] + NlEE[l1])*(ClBBrec[l2] + NlBB[l2])
        denom   = NlEE[l1] * NlBB[l2]
        sumell += gaEB / denom
    return (2.*l0List+1.) / sumell

def bispectrum_Tdsc_Tsc_Tsc(fsky, l0Max, ClTT, Cltautaurei, NICLdscdsc, NICLscsc):
    l1List = np.arange(2, l0Max)
    l2List = np.arange(2, l0Max)

    all_possible_pairs = np.asarray(list(itertools.product(l1List, l2List)))

    sumlist = 0.
    for ind, (l1,l2) in enumerate(all_possible_pairs):
        wig3j = wigner.wigner_3jj(l2, l1, 0,  0)

        am, bm  = max(2, int(wig3j[0])), min(int(wig3j[1]), l0Max-1)
        l0list  = np.arange(am, bm+1)
        aw, bw  = int(am - wig3j[0]), int(bm - wig3j[0])
        wig000  = np.abs(wig3j[2][aw:bw+1])**2.

        norm   = wig000 * (2.*l0list+1.) * (2.*l1+1.) * (2.*l2+1.) / (4. * np.pi)
        numer  = norm * ((ClTT[l0list] + ClTT[l1]) * Cltautaurei[l2])**2.
        denom  = NICLdscdsc[l2] * NICLscsc[l0list] * NICLscsc[l1]

        sumlist += np.sum(numer / denom)
    # extra factor of 1/2 inside the square root comes from symmetrization wrt Tsc Tsc fields
    return 0.76 / (fsky * sumlist / 2.)**0.25

def bispectrum_Tdsc_Esc_Bsc(fsky, l0Max, ClEE, Cltautaurei, NICLdscdsc, NICLEEscsc, NICLBBscsc):
    l1List = np.arange(2, l0Max)
    l2List = np.arange(2, l0Max)

    all_possible_pairs = np.asarray(list(itertools.product(l1List, l2List)))

    sumlist = 0.
    for ind, (l1,l2) in enumerate(all_possible_pairs):
        wig3j = wigner.wigner_3jj(l2, l1, 0,  2)

        am, bm  = max(2, int(wig3j[0])), min(int(wig3j[1]), l0Max-1)
        l0list  = np.arange(am, bm+1)
        aw, bw  = int(am - wig3j[0]), int(bm - wig3j[0])
        wig220  = np.abs(wig3j[2][aw:bw+1])**2.

        Jell   = l0list+l1+l2
        delto  = 0.5*(1. - (-1.)**Jell)

        norm   = delto * wig220 * (2.*l0list+1.) * (2.*l1+1.) * (2.*l2+1.) / (4. * np.pi)
        numer  = norm * ((ClEE[l0list] + ClEE[l1]) * Cltautaurei[l2])**2.
        denom  = NICLdscdsc[l2] * NICLBBscsc[l0list] * NICLEEscsc[l1]

        sumlist += np.sum(numer / denom)
    return 0.76 / (fsky * sumlist)**0.25


def w000(Ell, ell0, ell1, ell2):
    # very fast wigner 3j with m1 = m2 = m3 = 0
    g = Ell/2.
    w = np.exp(0.5*(lgamma(2.*g-2.*ell0+1.)+lgamma(2.*g-2.*ell1+1.)+lgamma(2.*g-2.*ell2+1.)-lgamma(2.*g+2.))\
                          +lgamma(g+1.)-lgamma(g-ell0+1.)-lgamma(g-ell1+1.)-lgamma(g-ell2+1.))
    return w * (-1.)**g


########## Covariance Matrices + Fischer forecasting ###########

def sigma_screening(epsilon4, fsky, ellmin, ellmax, screening, leftover):
    ClTTNl = epsilon4 * screening[0, :ellmax] + leftover[0, :ellmax]
    ClEENl = epsilon4 * screening[1, :ellmax] + leftover[1, :ellmax]
    ClBBNl = epsilon4 * screening[2, :ellmax] + leftover[2, :ellmax]

    dClTTde4 = screening[0, :ellmax]
    dClEEde4 = screening[1, :ellmax]
    dClBBde4 = screening[2, :ellmax]

    TrF = np.zeros(ellmax)
    for el in range(ellmin, ellmax):
        CCov    = np.diag([ClTTNl[el], ClEENl[el], ClBBNl[el]])
        CCovInv = np.linalg.inv(CCov)
        dCovde4 = np.diag([dClTTde4[el], dClEEde4[el], dClBBde4[el]])
        TrF[el] = 0.5*(2.*el+1.)*np.trace(CCovInv@dCovde4@CCovInv@dCovde4)
    return 0.7 / (fsky * np.sum(TrF))**0.125

def sigma_screening_TT(epsilon4, fsky, ellmin, ellmax, screening, leftover):
    #print('Full ', np.shape(leftover), np.shape(screening))
    ClTTNl = epsilon4 * screening[0, :ellmax] + leftover[0, :ellmax]
    ClEENl = epsilon4 * screening[1, :ellmax] + leftover[1, :ellmax]
    ClBBNl = epsilon4 * screening[2, :ellmax] + leftover[2, :ellmax]

    dClTTde4 = screening[0, :ellmax]
    dClEEde4 = screening[1, :ellmax]
    dClBBde4 = screening[2, :ellmax]

    TrF = np.zeros(ellmax)
    for el in range(ellmin, ellmax):
        CCovInv = 1./ClTTNl[el]
        dCovde4 = dClTTde4[el]
        TrF[el] = 0.5*(2.*el+1.) * (CCovInv*dCovde4*CCovInv*dCovde4)
    return 0.7 / (fsky * np.sum(TrF))**0.125

def sigma_screeningVtemplate(TCMB, ep2, fsky, ellmin, ellmax, cltauscreening, leftover, templategal):
    ClTTNl      = ep2**2.* cltauscreening[0,:ellmax] + leftover[0,:ellmax]
    Clττ        = templategal[:ellmax]
    ClTτscr     = ep2    * templategal[:ellmax] * TCMB
    dClTTde2    = 2.*ep2 * cltauscreening[0,:ellmax]
    dClTτscrde2 = templategal[:ellmax] * TCMB

    TrF = np.zeros(ellmax)
    for el in range(ellmin, ellmax):
        CCov = np.asarray([[Clττ[el]   , ClTτscr[el]],\
                           [ClTτscr[el], ClTTNl[el] ]])
        CCovInv = np.linalg.inv(CCov)
        dCovde2 = np.asarray([[0.             , dClTτscrde2[el]],\
                              [dClTτscrde2[el], dClTTde2[el]   ]])

        TrF[el] = 0.5*(2.*el+1.)*np.trace(CCovInv@dCovde2@CCovInv@dCovde2)
    return 0.76 / (fsky * np.sum(TrF))**0.25



def battagliaAGN(m200, zs):
    # power law fits:
    rho0  = 4000. * (m200/1e14)**0.29    * (1.+zs)**(-0.66)
    alpha = 0.88  * (m200/1e14)**(-0.03) * (1.+zs)**0.19
    beta  = 3.83  * (m200/1e14)**0.04    * (1.+zs)**(-0.025)
        
    gamma = -0.2
    xc    = 0.5
    return rho0, alpha, beta, gamma, xc

def battagliaSH(m200, zs):
    # power law fits:
    rho0  = 1.9e4 * (m200/1e14)**0.09     * (1.+zs)**(-0.95)
    alpha = 0.7   * (m200/1e14)**(-0.017) * (1.+zs)**0.27
    beta  = 4.43  * (m200/1e14)**0.005    * (1.+zs)**0.037

    gamma = -0.2
    xc    = 0.5
    return rho0, alpha, beta, gamma, xc

def bestfitACT():
    rho0  = np.exp(2.6*np.log(10))
    alpha = 1.
    beta  = 2.6
    gamma = -0.2
    xc    = 0.6
    return rho0, alpha, beta, gamma, xc

def limber_int(ells,zs,ks,Pzks,hzs,chis):
    hzs = np.array(hzs).reshape(-1)
    chis = np.array(chis).reshape(-1)
    prefactor = hzs / chis**2.

    f = interp2d(ks, zs, Pzks, bounds_error=True)     

    Cells = np.zeros(ells.shape)
    for ii, ell in enumerate(ells):
        kevals = (ell+0.5)/chis

        # hack suggested in https://stackoverflow.com/questions/47087109/evaluate-the-output-from-scipy-2d-interpolation-along-a-curve
        # to get around scipy.interpolate limitations
        interpolated = si.dfitpack.bispeu(f.tck[0], f.tck[1], f.tck[2], f.tck[3], f.tck[4], kevals, zs)[0]

        Cells[ii] = np.trapz(interpolated*prefactor, zs)
    return Cells
