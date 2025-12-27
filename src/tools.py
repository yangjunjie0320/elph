import numpy as np
import scipy, time, os, h5py
from pyscf import lib
from pyscf.pbc import gto, dft
from pyscf.pbc.tools import k2gamma
# from gwpt.tools.eph import get_map3, get_map3_single_k

import line_profiler

from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.tools.k2gamma import get_phase


def get_kconserv2(pcell, km, qm=None):
    """
    pcell is the primitive cell; km and qm are two
    sets of k-points meshes; qm can be None, in which
    case qm = km; and will fall to the original function
    in pyscf.pbc.lib.kpts_helper.get_kconserv_ria.

    Given an array of indices, i = kconserv[ik, iq],
    which satisfies the momentum conservation condition,
        (km[ik] + qm[iq] - km[i]) dot a = 2 * n * pi
    where a is the lattice vectors.
    """
    if qm is None:
        return kpts_helper.get_kconserv_ria(pcell, km)

    nk = len(km)
    nq = len(qm)
    a = pcell.lattice_vectors() / (2 * np.pi)
    kconserv = np.full((nk, nq), np.nan, dtype=float)

    kk = km[:, None, :] + qm[None, :, :]
    assert kk.shape == (nk, nq, 3)

    for ik in range(nk):
        x = lib.einsum("wx,abx->wab", a, kk - km[ik])
        x_int = np.rint(x)
        mask = np.einsum("wab->ab", abs(x - x_int)) < 1e-9
        print(mask.shape, kconserv.shape)
        kconserv[mask] = ik

    message = "km and qm are not compatible"
    assert not np.isnan(kconserv).any(), message

    return kconserv.astype(int)


from pyscf.eph.rhf import CUTOFF_FREQUENCY, KEEP_IMAG_FREQUENCY
from pyscf.data.nist import HARTREE2WAVENUMBER, MP_ME


def eigh_hess(h_q, freq_cutoff=100, keep_imag_freq=False, verbose=4):
    log = lib.logger.new_logger(None, verbose)

    nx = h_q[0].shape[1] * h_q[0].shape[2]

    coeff_mode_q = []
    freq_mode_q = []

    for hq in h_q:
        hq = hq.reshape(nx, nx)
        mask = abs(hq) < 1e-12
        hq[mask] *= 0.0

        e, c = scipy.linalg.eigh(hq)
        real_ix = np.where(e > 0)[0]
        imag_ix = np.where(e < 0)[0]

        freq_real_in_au = abs(e[real_ix]) ** 0.5
        freq_imag_in_au = abs(e[imag_ix]) ** 0.5
        freq_real_in_wn = freq_real_in_au * HARTREE2WAVENUMBER
        freq_imag_in_wn = freq_imag_in_au * HARTREE2WAVENUMBER

        log.info("\n** Real Eigenmodes (in cm-1) **")
        for i, iw in enumerate(freq_real_in_wn):
            is_filtered = iw < freq_cutoff
            message = f"Mode {i:2d} Omega = {iw:6.4f}"
            if is_filtered:
                message += " (filtered)"
            log.info(message)

        if len(freq_imag_in_wn) > 0:
            log.info("\n** Imaginary Eigenmodes (in cm-1) **")

        for i, iw in enumerate(freq_imag_in_wn):
            is_filtered = iw < freq_cutoff
            message = f"Mode {i:2d} Omega = {iw:6.4f} j"
            log.info(message)

        freq = None
        mode = None

        if keep_imag_freq:
            freq = np.concatenate((-freq_imag_in_au, freq_real_in_au))
            mode = np.concatenate((mode[:, imag_ix], mode[:, real_ix]), axis=1)
        else:
            mask = freq_real_in_wn > freq_cutoff
            freq = freq_real_in_au[mask]
            mode = c[:, real_ix][:, mask]

        freq_in_wn = freq * HARTREE2WAVENUMBER
        if len(freq_in_wn) < nx:
            log.info("\n** Remaining Eigenmodes (in cm-1) **")
            for i, iw in enumerate(freq_in_wn):
                message = f"Mode {i:2d} Omega = {iw:6.4f}"
                if iw.imag == 0:
                    message += " (real)"
                else:
                    message += " (imag)"
                log.info(message)

        coeff_mode_q.append(mode)
        freq_mode_q.append(freq)

    return freq_mode_q, coeff_mode_q


def dg_g2k(dg, pcell=None, kc=None, qc=None):
    nq = len(qc)
    phase = get_phase(pcell, kpts=qc)[1]

    natm = pcell.natm
    dg = dg.reshape(natm * 3, nq, natm * 3)
    h_q = lib.einsum("Rq,xRy->qxy", phase, dg)
    h_q = h_q.reshape(nq, natm, 3, natm, 3) * np.sqrt(nq)

    return h_q


def dv_g2k(dv, pcell=None, kc=None, qc=None):
    km = k2gamma.kpts_to_kmesh(pcell, kc)
    ktv = k2gamma.translation_vectors_for_kmesh(pcell, km)
    nk = len(kc)

    qm = k2gamma.kpts_to_kmesh(pcell, qc)
    qtv = k2gamma.translation_vectors_for_kmesh(pcell, qm)
    dqtv = qtv - qtv[0]
    nq = len(qc)

    nao = pcell.nao_nr()
    nx = pcell.natm * 3

    assert dv.shape == (nx, nq, nao, nq, nao)
    g_qk_xmn = []
    for q, vq in enumerate(qc):
        f1 = np.exp(1j * np.dot(dqtv, (kc + vq).T)).conj()
        f2 = np.exp(1j * np.dot(dqtv, kc.T))
        g_qk_xmn.append(lib.einsum("xRmSn,Rk,Sk->kxmn", dv, f1, f2))

    g_qk_xmn = np.array(g_qk_xmn).reshape(nq, nk, nx, nao, nao)
    return g_qk_xmn
