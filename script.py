#!/usr/bin/env python3.7.16

# To run this script, in a separate terminal type:
#### conda activate conda_env
#### python3 script.py >> ./data/output.txt

import os,sys
#sys.path.remove('/home/dpirvu/DarkPhotonxunWISE/hmvec-master')
sys.path.append('/home/dpirvu/DarkPhoton/hmvec-master/')
sys.path.append('/home/dpirvu/python_stuff/')
print([ii for ii in sys.path])
import hmvec as hm
import numpy as np

from compute_power_spectra import *
from params import *
from plotting import *

import functools
from concurrent.futures import ProcessPoolExecutor

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# Select electron profile
conv_gas  = False
conv_NFW  = True
pick_prof = (True if conv_gas else False)

# Select which optical depth PS to compute
compute_dark    = True
compute_thomson = False
compute_cross   = False

# Select which CMB isotropic screening to compute
compute_full_screening = False
screening_compute_dark = False
screening_compute_cross = False
screening_compute_thomson = False

# Compute bispectrum forecats?
compute_bispect = False

# Select DP mass
maind=0

ellMax = 7000
ells   = np.arange(ellMax)

zthr  = 6.
zreio = 6.

if conv_gas:
    MA = dictKey_gas[maind]
    zMin, zMax, rMin, rMax = chooseModel(MA, modelParams_gas)
    name = 'battagliaAGN'
    rscale = False

elif conv_NFW:
    MA = dictKey_NFW[maind]
    zMin, zMax, rMin, rMax = chooseModel(MA, modelParams_NFW)
    name = None
    rscale = True

zMax = min(zthr, zMax)
nZs  = 50
ms  = np.geomspace(1e11,1e17,100)       # masses
zs  = np.linspace(zMin, zMax,nZs)       # redshifts
rs  = np.linspace(rMin, rMax,100000)    # halo radius
ks  = np.geomspace(1e-4,1e3, 1001)      # wavenumbers

# If parallelized 
num_workers = 20
chunksize = max(1, len(ells)//num_workers)

print('DARK PHOTON MASS:', MA)
print('Redshifts:', zMin, zMax, nZs)

# Halo Model
dictnumber= 21
unWISEcol = 'blue'
pathdndz  = "/home/dpirvu/DarkPhotonxunWISE/dataHOD/normalised_dndz_cosmos_0.txt"
hod_name  = "unWISE"+unWISEcol

hcos = hm.HaloModel(zs, ks, ms=ms, mass_function='tinker', mdef='vir', concmode='duffy')#, unwise_color=unWISEcol, choose_dict=dictnumber)
#hcos.add_hod(name=hod_name)
print('Test hmvec')
print(hm.default_params['H0'])
print(hcos.conc)

chis     = hcos.comoving_radial_distance(zs)
rvirs    = hcos.rvir(ms[None,:],zs[:,None])
cs       = hcos.concentration()
Hz       = hcos.h_of_z(zs)
nzm      = hcos.get_nzm()
biases   = hcos.get_bh()
deltav   = hcos.deltav(zs)
rhocritz = hcos.rho_critical_z(zs)
m200c, r200c = get_200critz(zs, ms, cs, rhocritz, deltav)

path_params = np.asarray([MA, nZs, zMin, zMax, ellMax, rscale])
path_params_thom = np.asarray([nZs, zMin, zreio, ellMax, rscale])

# Milky Way stuff: zMW=0.
HMW  = hcos.h_of_z(0.)
rsMW = np.geomspace(1e-4,10,10000)  # halo radius
deltavMW = hcos.deltav(0.)[0]
rhocritzMW = hcos.rho_critical_z(0.)

print('Importing CMB power spectra and adding temperature monopole.')
CMB_ps        = hcos.CMB_power_spectra()
unlenCMB      = CMB_ps['unlensed_scalar']
unlenCMB      = unlenCMB[:ellMax, :]
unlenCMB[0,0] = TCMB**2.
lensedCMB     = CMB_ps['lensed_scalar']
lensedCMB     = lensedCMB[:ellMax, :]
lensedCMB[0,0]= TCMB**2.

dvols  = get_volume_conv(chis, Hz)
PzkLin = hcos._get_matter_power(zs, ks, nonlinear=False)
Pzell  = get_fourier_to_multipole_Pkz(zs, ks, chis, ellMax, PzkLin)
Pzell0 = Pzell.transpose(1,0)
print('Done turning into multipoles.')

if compute_dark:
    print('Computing MW stuff.')
    rcrossMW, probMW = dark_photon_conv_prob_MilkyWay(mMWvir, rMWvir, rhocritzMW, deltavMW, csMW, HMW, rEarth, rsMW, MA, pick_prof, name=name)

    print('Computing crossing radii.')
    rcross = get_rcross_per_halo(zs, ms, rs, rvirs, rhocritz, deltav, cs, MA, pick_prof, name=name)
    ucosth, angs = get_halo_skyprofile(zs, chis, rcross)

    print('Computing multipole expansion of angular probability u.')
    partial_u = functools.partial(get_uell0, angs, ucosth)
    with ProcessPoolExecutor(num_workers) as executor:
        uell0 = list(executor.map(partial_u, ells, chunksize=chunksize))

    print('Computing probability to convert.')
    prob = dark_photon_conv_prob(zs, ms, rs, rvirs, rhocritz, deltav, cs, MA, rcross, pick_prof, rscale=rscale, name=name)

    prob00 = prob * uell0[0]
    avtau, dtaudz = get_avtau(zs, ms, nzm, dvols, prob00)

    if conv_gas:
        np.save(MW_data_path_gas(*path_params), [rcrossMW, probMW])
        np.save(data_path_gas(*path_params),    [rcross, prob, avtau, dtaudz, uell0])
    elif conv_NFW:
        np.save(MW_data_path(*path_params), [rcrossMW, probMW])
        np.save(data_path(*path_params),    [rcross, prob, avtau, dtaudz, uell0])

    if conv_gas:
        rcrossMW, probMW = np.load(MW_data_path_gas(*path_params))
        rcross, prob, avtau, dtaudz, uell0 = np.load(data_path_gas(*path_params))
    elif conv_NFW:
        rcrossMW, probMW = np.load(MW_data_path(*path_params))
        rcross, prob, avtau, dtaudz, uell0 = np.load(data_path(*path_params))

    ct = np.sqrt((4.*np.pi)/(2*ells+1))
    zell_tau = (prob[None,...] * uell0) * ct[:, None, None]

    # Assemble power spectra
    int_uell_1h = np.trapz(nzm[None,...] * zell_tau**2.               , ms, axis=-1)
    int_uell_2h = np.trapz(nzm[None,...] * biases[None,...] * zell_tau, ms, axis=-1)

    Cl1h  = np.trapz(dvols[None,:] * int_uell_1h                     , zs, axis=1)
    Cl2h  = np.trapz(dvols[None,:] * np.abs(int_uell_2h)**2. * Pzell0, zs, axis=1)
    scrTT0= (Cl1h + Cl2h) * TCMB**2.

    if conv_gas:
        np.save(cl_data_tautau_path_gas(*path_params), [Cl1h, Cl2h, scrTT0])
    elif conv_NFW:
        np.save(cl_data_tautau_path(*path_params), [Cl1h, Cl2h, scrTT0])


if compute_thomson:
    print('Compute probability to Thomson scatter and angular dependence.')
    prob_thom = get_thomsontau_per_halo(zs, ms, ellMax, chis, rvirs, rhocritz, deltav, cs, MA, name=name)
    np.save(cl_data_thomthom_prob_path(*path_params_thom), prob_thom)

    ct = np.sqrt((4.*np.pi)/(2*ells+1))
    zell_thom = prob_thom * ct[:, None, None]

    # Assemble power spectra
    int_uthell_1h = np.trapz(nzm[None,...] * zell_thom**2.               , ms, axis=-1)
    int_uthell_2h = np.trapz(nzm[None,...] * biases[None,...] * zell_thom, ms, axis=-1)

    Clth1h  = np.trapz(dvols[None,:] * int_uthell_1h                     , zs, axis=1)
    Clth2h  = np.trapz(dvols[None,:] * np.abs(int_uthell_2h)**2. * Pzell0, zs, axis=1)
    np.save(cl_data_thomthom_path(*path_params_thom), np.array([Clth1h, Clth2h]))


if compute_cross:
    print('Importing data.')
    if conv_gas:   rcross, prob, avtau, dtaudz, uell0 = np.load(data_path_gas(*path_params))
    elif conv_NFW: rcross, prob, avtau, dtaudz, uell0 = np.load(data_path(*path_params))
    prob_thom = np.load(cl_data_thomthom_prob_path(*path_params_thom))

    ct = np.sqrt((4.*np.pi)/(2*ells+1))
    zell_tau  = (prob[None,...] * uell0) * ct[:, None, None]
    zell_thom =  prob_thom * ct[:, None, None]

    print('Computing Thomson x Dark Photon 1-halo angular PS.')
    int_zell_tauth= np.trapz(nzm[None,...] * zell_tau * zell_thom, ms, axis=-1)
    int_zell_thom = np.trapz(nzm[None,...] * biases[None,...] * zell_thom, ms, axis=-1)
    int_zell_tau  = np.trapz(nzm[None,...] * biases[None,...] * zell_tau , ms, axis=-1)

    Cell_tauth_1h  = np.trapz(dvols[None,:] * int_zell_tauth                       , zs, axis=1)
    Cell_tauth_2h  = np.trapz(dvols[None,:] * int_zell_tau * int_zell_thom * Pzell0, zs, axis=1)

    if conv_gas:   np.save(cl_data_thomtau_path_gas(*path_params), [Cell_tauth_1h, Cell_tauth_2h])
    elif conv_NFW: np.save(cl_data_thomtau_path(*path_params),     [Cell_tauth_1h, Cell_tauth_2h])


if compute_full_screening:
    print('Importing data.')
    if conv_gas:
        Cl1h, Cl2h, Celltautau = np.load(cl_data_tautau_path_gas(*path_params))
        thom_Cl1h, thom_Cl2h, _ = np.load(cl_data_thomtau_path_gas(*path_params))

    elif conv_NFW:
        Cl1h, Cl2h, Celltautau = np.load(cl_data_tautau_path(*path_params))
        thom_Cl1h, thom_Cl2h, _ = np.load(cl_data_thomtau_path(*path_params))

    thomthom_Cl1h, thomthom_Cl2h = np.load(cl_data_thomthom_path(*path_params_thom))


    if screening_compute_dark:
        print('Computing new CMB dark screening PS.')
        Celltautau = Cl1h + Cl2h # this is because I compute the monopole contribution separately; Celltautau should not have the factor of TCMB**2.
        llist, scrTT, scrEE, scrBB, scrTE = get_scrCLs(ellMax, CMB=unlenCMB, DPCl=Celltautau)

        if conv_gas:   np.save(fullscr_tautau_path_gas(*path_params), [llist, scrTT, scrEE, scrBB, scrTE])
        elif conv_NFW: np.save(fullscr_tautau_path(*path_params),     [llist, scrTT, scrEE, scrBB, scrTE])

    if screening_compute_cross:
        print('Computing new CMB thomson cross dark screening PS.')
        Cellthomtau = thom_Cl1h + thom_Cl2h
        llist, scrTT, scrEE, scrBB, scrTE = get_scrCLs(ellMax, CMB=unlenCMB, DPCl=Cellthomtau)

        if conv_gas:   np.save(fullscr_thomtau_path_gas(*path_params), [llist, scrTT, scrEE, scrBB, scrTE])
        elif conv_NFW: np.save(fullscr_thomtau_path(*path_params),     [llist, scrTT, scrEE, scrBB, scrTE])

    if screening_compute_thomson:
        print('Computing new CMB thomson screening PS.')
        Cellthomthom = thomthom_Cl1h + thomthom_Cl2h
        llist, scrTT, scrEE, scrBB, scrTE = get_scrCLs(3000, CMB=unlenCMB, DPCl=Cellthomthom)
        np.save(fullscr_thomthom_path(*path_params_thom), [llist, scrTT, scrEE, scrBB, scrTE])


if compute_bispect:
    print("Importing data:")



    for eind, (expname, experiment) in enumerate(zip(['Planck', 'CMBS4', 'CMBHD'], [Planck, CMBS4, CMBHD])):
        if expname == 'Planck': 
            mmm = 3000
        else:
            mmm = 6000

        fsky = [0.7, 0.5, 0.5][eind]
        baseline = ghztoev(30)
        units = xov(baseline) / baseline

        Cellthomtau = thom_Cl1h + thom_Cl2h
        screening_cross = Cellthomtau * units * TCMB

        print('Computing bispectrum for ', expname)
        ILCnoise = np.load(ILCnoisePS_path_gas(MA, nZs, zMin, zreio, ellMax, expname))
        NTTdscdsc = ILCnoise[0, :mmm]

        BB_ILCnoise = np.load(BB_ILCnoisePS_path_gas(expname, zreio))
        NTTscsc = BB_ILCnoise[0, :mmm]
        NEEscsc = BB_ILCnoise[1, :mmm]
        NBBscsc = BB_ILCnoise[2, :mmm]

        ClTT = lensedCMB.T[0, :mmm]
        ClEE = lensedCMB.T[1, :mmm]

        bispTTT = bispectrum_Tdsc_Tsc_Tsc(fsky, mmm, ClTT, screening_cross, NTTdscdsc, NTTscsc)
        np.save(path_bispTdscTscTsc_gas(*path_params, expname), bispTTT)

        bispTEB = bispectrum_Tdsc_Esc_Bsc(fsky, mmm, ClEE, screening_cross, NTTdscdsc, NEEscsc, NBBscsc)
        np.save(path_bispTdscEscBsc_gas(*path_params, expname), bispTEB)
        print(expname, 'Done!')

print('All Done.')
