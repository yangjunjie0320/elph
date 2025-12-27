import numpy as np
import scipy.linalg

from pyscf import lib
from pyscf.data.nist import HARTREE2WAVENUMBER
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.tools.k2gamma import get_phase


def get_k_plus_q(pcell, kpts_k, kpts_q=None):
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
    if kpts_q is None:
        return kpts_helper.get_kconserv_ria(pcell, kpts_k)

    nk = len(kpts_k)
    nq = len(kpts_q)
    a = pcell.lattice_vectors()
    table = np.full((nk, nq), np.nan, dtype=float)

    k_plus_q = kpts_k[:, None, :] + kpts_q[None, :, :]
    assert k_plus_q.shape == (nk, nq, 3)

    for k, vk in enumerate(kpts_k):
        d = k_plus_q - vk
        a_dot_d = lib.einsum("wx,abx->wab", a, d)
        a_dot_d /= 2 * np.pi

        m = np.sum(abs(a_dot_d - np.rint(a_dot_d)), axis=0) < 1e-9
        table[k, m] = k

    message = "kpts_k and kpts_q are not compatible"
    assert not np.isnan(table).any(), message

    return table.astype(int)


def eigh_hess(hess_q, mass, freq_cutoff=100, keep_imag_freq=False, verbose=4):
    log = lib.logger.Logger(verbose)

    sqrt_mm = np.sqrt(mass[:, None] * mass[None, :])
    nx = hess_q[0].shape[1] * hess_q[0].shape[2]
    freq_q = []
    coeff_q = []

    assert not keep_imag_freq

    for hq in hess_q:
        # Diagonalize Hessian matrix
        hq = hq / sqrt_mm[None, :, None, :, None]
        hq = hq.reshape(nx, nx)
        hq[abs(hq) < 1e-12] = 0.0

        eigval, eigvec = scipy.linalg.eigh(hq)
        mask = eigval > 0

        # Convert eigenvalues to frequencies
        freq_real_au = np.sqrt(eigval[mask])
        freq_real_wn = freq_real_au * HARTREE2WAVENUMBER

        fq = freq_real_au
        cq = eigvec[:, mask]

        # Log real frequencies
        if len(freq_real_wn) > 0:
            log.info("\n** Real Eigenmodes (in cm-1) **")
            for i, w in enumerate(freq_real_wn):
                is_filtered = "(filtered)" if w < freq_cutoff else ""
                log.info(f"Mode {i:2d} Omega = {w:6.4f} {is_filtered}")

        mask = fq > freq_cutoff
        fq = fq[mask]
        cq = cq[:, mask]

        sqrt_mf = np.sqrt(2 * mass[:, None] * fq)
        cq = cq / sqrt_mf[:, None, :]

        freq_q.append(fq)
        coeff_q.append(cq)

    return freq_q, coeff_q


def dg_g2k(dg, pcell=None, kpts_k=None, kpts_q=None):
    nq = len(kpts_q)
    phase = get_phase(pcell, kpts=kpts_q)[1]

    natm = pcell.natm
    nx = natm * 3

    dg = dg.reshape(nx, nq, nx)
    hess_q = lib.einsum("Rq,xRy->qxy", phase, dg)
    hess_q = hess_q.reshape(nq, natm, 3, natm, 3)
    return hess_q * np.sqrt(nq)


def dv_g2k(dv, pcell=None, kpts_k=None, kpts_q=None):
    km = k2gamma.kpts_to_kmesh(pcell, kpts_k)
    ktv = k2gamma.translation_vectors_for_kmesh(pcell, km)
    nk = len(kpts_k)

    qm = k2gamma.kpts_to_kmesh(pcell, kpts_q)
    qtv = k2gamma.translation_vectors_for_kmesh(pcell, qm)
    dqtv = qtv - qtv[0]
    nq = len(kpts_q)

    nao = pcell.nao_nr()
    nx = pcell.natm * 3

    assert dv.shape == (nx, nq, nao, nq, nao)
    g_qk_xmn = []
    for q, vq in enumerate(kpts_q):
        f1 = np.exp(1j * np.dot(dqtv, (kpts_k + vq).T))
        f2 = np.exp(1j * np.dot(dqtv, kpts_k.T))
        g_qk_xmn.append(lib.einsum("xRmSn,Rk,Sk->kxmn", dv, f1.conj(), f2))

    g_qk_xmn = np.array(g_qk_xmn).reshape(nq, nk, nx, nao, nao)
    return g_qk_xmn
