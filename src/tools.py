import numpy
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
    assert nk >= nq

    lv = pcell.lattice_vectors()
    table = numpy.full((nk, nq), numpy.nan, dtype=float)

    k_plus_q = kpts_k[:, None, :] + kpts_q[None, :, :]
    assert k_plus_q.shape == (nk, nq, 3)

    for k, vk in enumerate(kpts_k):
        lv_dot_dk = lib.einsum("wx,abx->wab", lv, k_plus_q - vk)
        lv_dot_dk = lv_dot_dk / 2 / numpy.pi

        d = abs(lv_dot_dk - numpy.rint(lv_dot_dk))
        m = numpy.sum(d, axis=0) < 1e-9
        table[m] = k

    message = "kpts_k and kpts_q are not compatible"
    assert not numpy.isnan(table).any(), message

    return table.astype(int)


def eigh_hess(hess_q, mass, freq_cutoff=100, keep_imag_freq=False, verbose=4):
    log = lib.logger.Logger(verbose)

    sqrt_mm = numpy.sqrt(mass[:, None] * mass[None, :])
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
        freq_real_au = numpy.sqrt(eigval[mask])
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

        sqrt_mf = numpy.sqrt(2 * mass[:, None] * fq)
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
    return hess_q * numpy.sqrt(nq)


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
        f1 = numpy.exp(1j * numpy.dot(dqtv, (kpts_k + vq).T))
        f2 = numpy.exp(1j * numpy.dot(dqtv, kpts_k.T))
        g_qk_xmn.append(lib.einsum("xRmSn,Rk,Sk->kxmn", dv, f1.conj(), f2))

    g_qk_xmn = numpy.array(g_qk_xmn).reshape(nq, nk, nx, nao, nao)
    return g_qk_xmn


if __name__ == "__main__":
    from pyscf import pbc

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
    cell.unit = "B"
    cell.verbose = 4
    cell.build()

    lv = cell.lattice_vectors()
    kpts_k = cell.make_kpts([6, 6, 6])
    kpts_q = cell.make_kpts([3, 3, 3])
    k_plus_q = get_k_plus_q(cell, kpts_k, kpts_q)

    for q, vq in enumerate(kpts_q):
        for k1, vk1 in enumerate(kpts_k):
            k2 = k_plus_q[k1, q]
            vk2 = kpts_k[k2]

            dk = vk1 + vq - vk2
            lv_dot_dk = lib.einsum("wx,x->w", lv, dk)
            lv_dot_dk = lv_dot_dk / 2 / numpy.pi

            err = abs(lv_dot_dk - numpy.rint(lv_dot_dk)).max()
            assert err < 1e-9, f"Error: {err:.6e}"
