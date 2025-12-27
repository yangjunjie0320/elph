"""
Electron-phonon coupling calculation module.

This module provides functionality for calculating electron-phonon coupling
using PySCF for periodic systems.
"""

import sys, os
from dataclasses import dataclass

import numpy

from pyscf import lib, pbc
from pyscf.data.nist import MP_ME
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.df import FFTDF
from pyscf.pbc.dft.multigrid import MultiGridNumInt2
from pyscf.pbc.dft.numint import NumInt
from pyscf.pbc.tools.k2gamma import k2gamma, kpts_to_kmesh

from tools import dg_g2k, dv_g2k, eigh_hess, get_k_plus_q


@dataclass
class ElectronPhononCouplingInfo:
    # Input objects
    mf_qc: pbc.dft.KRKS
    mf_kc: pbc.dft.KRKS
    pcell: pbc.gto.Cell

    # k-point mesh information
    kpts_q: numpy.ndarray
    kpts_k: numpy.ndarray
    mesh_q: numpy.ndarray
    mesh_k: numpy.ndarray

    # Phonon information
    hess_q: numpy.ndarray
    freq_mode_q: numpy.ndarray
    coeff_mode_q: numpy.ndarray
    num_mode: int

    # Electronic information
    ene_band_k: numpy.ndarray
    coeff_band_k: numpy.ndarray
    num_band: int

    # Electron-phonon coupling matrices
    g_qk_xmn: numpy.ndarray
    g_qk_lmn: numpy.ndarray

    # Derivative information
    dv: numpy.ndarray
    dg: numpy.ndarray


def fd_eph(kmf, disp=1e-4, disp_full_supercell=False, verbose=4):
    log = logger.new_logger(kmf, verbose)

    assert isinstance(kmf, pbc.dft.rks.KohnShamDFT)
    assert isinstance(kmf, pbc.scf.khf.KSCF)
    assert not disp_full_supercell

    xc_code = kmf.xc
    xc_type = kmf._numint._xc_type(xc_code)
    xc_type_table = {"LDA": 0, "GGA": 1}
    ao_deriv = xc_type_table.get(xc_type)
    assert ao_deriv is not None

    is_hyb_xc = kmf._numint.libxc.is_hybrid_xc(kmf.xc)
    assert not is_hyb_xc, "Hybrid XC functionals are not supported"

    kpts_q = kmf.kpts
    nq = len(kpts_q)

    pcell = kmf.cell.copy()
    mesh_q = kpts_to_kmesh(pcell, kpts_q)

    smf = k2gamma(kmf, mesh_q)
    scell = smf.cell.copy(deep=True)
    scell.verbose = verbose
    scell.build()

    natm_scell = scell.natm
    natm_pcell = pcell.natm
    nao_scell = scell.nao_nr()
    nao_pcell = pcell.nao_nr()
    nx = 3 * (natm_scell if disp_full_supercell else natm_pcell)

    smf.conv_tol = kmf.conv_tol
    smf._numint = MultiGridNumInt2(scell)
    smf.verbose = verbose

    # Sanity check
    assert isinstance(kmf.with_df, FFTDF)
    assert isinstance(smf.with_df, FFTDF)

    log.info("Running SCF for supercell")
    smf.kernel()

    dm0 = smf.make_rdm1()
    e_tot_pcell = kmf.e_tot * nq
    e_tot_scell = smf.e_tot
    error = abs(e_tot_pcell - e_tot_scell) / nq

    conv_tol = smf.conv_tol * nq
    if error > conv_tol:
        message = "Energy mismatch: % 6.4e > % 6.4e" % (error, conv_tol)
        message += "\nkmf use %s" % kmf._numint.__class__.__name__
        message += "\nsmf use %s" % smf._numint.__class__.__name__
        message += "\ne_tot_pcell = %12.8f" % e_tot_pcell
        message += "\ne_tot_scell = %12.8f" % e_tot_scell
        message += "\nConsider increasing ke_cutoff"
        log.warn(message)

    ni = NumInt()
    assert isinstance(ni, NumInt)

    hermi = 1

    mesh = scell.mesh
    coul_g = tools.get_coulG(scell, mesh=mesh)
    grid = smf.grids
    gx = grid.coords
    gw = grid.weights

    phi0 = ni.eval_ao(scell, gx, deriv=ao_deriv)

    def wv_phi(f, w):
        """Convert weight function to potential matrix."""
        f = f.real.copy()
        w = w.real.copy()

        mm = None
        ss = (0, scell.nbas)
        al = scell.ao_loc_nr()
        xc = xc_type

        v = ni._vxc_mat(scell, f, w, mm, xc, ss, al, 1)
        v = v + v.conj().swapaxes(-2, -1)
        return v.real

    def fd(a, x, sign, d=1e-3):
        """Calculate energy, gradient, and weight function for displacement."""
        x0 = scell.atom_coords()
        dx = numpy.zeros_like(x0)
        dx[a][x] = sign * d

        c = scell.copy(deep=True)
        c.set_geom_(x0 + dx, inplace=True)
        c.build()

        m = smf
        m.reset(c)
        m.kernel(dm0)
        assert m.converged

        e = m.e_tot
        g = m.nuc_grad_method().kernel()
        dm = m.make_rdm1()

        m._numint.vpplocG_part1 = None
        veff_ref = m.get_veff(dm=dm)

        gen_rho = ni._gen_rho_evaluator(m.cell, dm, hermi, False)[0]
        phi = ni.eval_ao(m.cell, gx, deriv=ao_deriv)
        rho = gen_rho(0, phi, None, xc_type)
        rho_g = tools.fft(rho, mesh)

        w = ni.eval_xc_eff(xc_code, rho, 1, None, xc_type, spin=0)[1]
        w = w + tools.ifft(coul_g * rho_g, mesh)
        w = w.real * gw

        veff_sol = wv_phi(phi, w)
        err = abs(veff_sol - veff_ref).max()
        if err > conv_tol:
            message = "veff mismatch: %6.4e > %6.4e" % (err, conv_tol)
            message += "\nbetween %s and %s" % (ni, m._numint)
            log.warn(message)

        return e, g, w

    dv = []
    dg = []

    for ix in range(nx):
        a, x = divmod(ix, 3)

        e_p, g_p, w_p = fd(a, x, +1, disp)
        e_m, g_m, w_m = fd(a, x, -1, disp)

        g_sol = (e_p - e_m) / (2 * disp)
        g_ref = g_p[a][x]
        err = abs(g_sol - g_ref)
        message = f"({a}, +{x}) g_sol = {g_sol:.6f}, g_ref = {g_ref:.6f}, err = {err:.6e}\n"

        g_ref = g_m[a][x]
        err = abs(g_sol - g_ref)
        message += f"({a}, -{x}) g_sol = {g_sol:.6f}, g_ref = {g_ref:.6f}, err = {err:.6e}\n"
        log.debug(message)

        dv.append(wv_phi(phi0, (w_p - w_m) / (2 * disp)))
        dg.append((g_p - g_m) / (2 * disp))

        log.info("ix = %d / %d" % (ix + 1, nx))

    dv = numpy.asarray(dv)
    dg = numpy.asarray(dg)
    dv = dv.reshape(natm_pcell * 3, nao_scell, nao_scell)
    dg = dg.reshape(natm_pcell * 3, natm_scell * 3)
    return dv, dg


def calculate_electron_phonon_coupling(
    mf_k=None,
    mf_q=None,
    disp=1e-4,
    verbose=4,
    freq_cutoff=100.0,
    keep_imag_freq=False,
):
    """
    Calculate electron-phonon coupling matrix elements.

    Parameters
    ----------
    mf_k : pbc.dft.KRKS
        Mean-field object for k-point mesh
    mf_q : pbc.dft.KRKS
        Mean-field object for q-point mesh
    disp : float, optional
        Displacement magnitude, by default 1e-4
    verbose : int, optional
        Verbosity level, by default 4
    freq_cutoff : float, optional
        Frequency cutoff in cm^-1, by default 100.0
    keep_imag_freq : bool, optional
        Whether to keep imaginary frequencies, by default False

    Returns
    -------
    info : ElectronPhononCouplingInfo
        Electron-phonon coupling information object containing all results
    """
    log = logger.new_logger(mf_k, verbose)

    pcell = mf_k.cell.copy()
    pcell.verbose = 10

    kpts_q = mf_q.kpts
    kpts_k = mf_k.kpts
    nq = len(kpts_q)
    nk = len(kpts_k)
    k_plus_q = get_k_plus_q(pcell, kpts_k, kpts_q)

    natm = pcell.natm
    nx = natm * 3
    nao = pcell.nao_nr()

    assert mf_q.converged
    assert mf_k.converged

    dv, dg = fd_eph(mf_q, disp=disp)
    dv = dv.reshape(natm * 3, nq, nao, nq, nao)
    dg = dg.reshape(natm * 3, natm * nq * 3)

    hess_q = dg_g2k(dg, pcell=pcell, kpts_q=kpts_q, kpts_k=kpts_k)
    g_qk_xmn = dv_g2k(dv, pcell=pcell, kpts_q=kpts_q, kpts_k=kpts_k)

    hess_q = hess_q.reshape(nq, natm, 3, natm, 3)
    g_qk_xmn = g_qk_xmn.reshape(nq, nk, nx, nao, nao)

    freq_mode_q, coeff_mode_q = eigh_hess(
        hess_q,
        mass=pcell.atom_mass_list() * MP_ME,
        freq_cutoff=freq_cutoff,
        keep_imag_freq=keep_imag_freq,
        verbose=verbose,
    )

    freq_mode_q = numpy.array(freq_mode_q).reshape(nq, -1)
    coeff_mode_q = numpy.array(coeff_mode_q).reshape(nq, nx, -1)
    num_mode = coeff_mode_q.shape[-1]

    ene_band_k = numpy.array(mf_k.mo_energy).reshape(nk, -1)
    coeff_band_k = numpy.array(mf_k.mo_coeff).reshape(nk, nao, -1)
    num_band = coeff_band_k.shape[-1]

    g_qk_lmn = []
    for q in range(nq):
        cq = coeff_mode_q[q]
        fq = freq_mode_q[q]
        for k1 in range(nk):
            ck1 = coeff_band_k[k1]
            k2 = k_plus_q[k1, q]
            ck2 = coeff_band_k[k2]

            scripts = "xmn,xl,mp,nq->lpq"
            operand = (g_qk_xmn[q, k1], cq, ck2.conj(), ck1)
            g_qk_lmn.append(lib.einsum(scripts, *operand))

    g_qk_lmn = numpy.array(g_qk_lmn)
    g_qk_lmn = g_qk_lmn.reshape(nq, nk, num_mode, num_band, num_band)

    mesh_q = kpts_to_kmesh(pcell, kpts_q)
    mesh_k = kpts_to_kmesh(pcell, kpts_k)

    # Create and return ElectronPhononCouplingInfo object
    info = ElectronPhononCouplingInfo(
        # Input objects
        mf_qc=mf_q,
        mf_kc=mf_k,
        pcell=pcell,
        # k-point mesh information
        kpts_q=kpts_q,
        kpts_k=kpts_k,
        mesh_q=mesh_q,
        mesh_k=mesh_k,
        # Phonon information
        hess_q=hess_q,
        freq_mode_q=freq_mode_q,
        coeff_mode_q=coeff_mode_q,
        num_mode=num_mode,
        # Electronic information
        ene_band_k=ene_band_k,
        coeff_band_k=coeff_band_k,
        num_band=num_band,
        # Electron-phonon coupling matrices
        g_qk_xmn=g_qk_xmn,
        g_qk_lmn=g_qk_lmn,
        # Derivative information
        dv=dv,
        dg=dg,
    )
    return info


if __name__ == "__main__":
    cell = pbc.gto.Cell()
    cell.atom = """
    C  0.000000000000   0.000000000000   0.000000000000
    C  1.685068664391   1.685068664391   1.685068664391
    """
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.a = """
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000
    """
    cell.unit = "B"  # Bohr
    cell.verbose = 0
    cell.ke_cutoff = 40
    cell.build()

    natm = cell.natm
    nao = cell.nao_nr()
    nx = 3 * natm

    qc_mesh = [1, 1, 3]
    qc = cell.make_kpts(qc_mesh)
    nqc = len(qc)

    xc = "PBE,PBE"
    mf_qc = pbc.dft.KRKS(cell, qc)
    mf_qc.xc = xc
    mf_qc.conv_tol = 1e-6
    mf_qc.conv_tol_grad = 1e-4
    mf_qc.verbose = 4
    mf_qc.kernel()

    kc_mesh = [6, 6, 6]
    kc = cell.make_kpts(kc_mesh)
    nkc = len(kc)

    mf_kc = pbc.dft.KRKS(cell, kc)
    mf_kc.xc = xc
    mf_kc.conv_tol = 1e-6
    mf_kc.conv_tol_grad = 1e-4
    mf_kc.verbose = 4
    mf_kc.kernel()
    info = calculate_electron_phonon_coupling(mf_kc, mf_qc, disp=1e-4)
