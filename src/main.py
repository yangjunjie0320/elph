from pyscf.dft import density_fit
import numpy, scipy

import pyscf
from pyscf import pbc, lib
from pyscf.lib import logger
from pyscf.pbc import tools

from pyscf.pbc.tools.k2gamma import k2gamma
from pyscf.pbc.df import FFTDF

from pyscf.pbc.dft.numint import NumInt, KNumInt
from pyscf.pbc.dft.multigrid import MultiGridNumInt2

from tools import eigh_hess, dg_g2k, dv_g2k
from tools import get_kconserv2
from pyscf.data.nist import MP_ME


@dataclass
class ElectronPhononCouplingInfo:
    mf_qc: pbc.dft.KRKS
    mf_kc: pbc.dft.KRKS
    pcell: pbc.gto.Cell

    # qc_list: numpy.ndarray
    # kc_list: numpy.ndarray
    # qc_plus_kc_list: numpy.ndarray
    kpt_q: numpy.ndarray
    kpt_k: numpy.ndarray

    # qc_mesh: numpy.ndarray
    # kc_mesh: numpy.ndarray
    mesh_q: numpy.ndarray
    mesh_k: numpy.ndarray

    dv_g: numpy.ndarray
    dg_g: numpy.ndarray

    hess_q: numpy.ndarray
    freq_mode_q: numpy.ndarray
    coeff_mode_q: numpy.ndarray
    coeff_band_k: numpy.ndarray
    num_mode: int
    num_band: int

    g_qk_xmn: numpy.ndarray
    g_qk_lmn: numpy.ndarray


def calculate_dv_dg(mf_qc=None, disp=1e-4):
    log = logger.Logger(kmf.stdout, kmf.verbose)
    self.disp = 1e-3
    self.disp_full_supercell = disp_full_supercell

    self.kmf = kmf
    self.verbose = kmf.verbose
    self.pcell = kmf.cell.copy()
    assert isinstance(kmf, pbc.dft.rks.KohnShamDFT)
    assert isinstance(kmf, pbc.scf.khf.KSCF)

    ni = kmf._numint
    xc_code = kmf.xc
    xc_type = ni._xc_type(xc_code)
    xc_type_table = {"LDA": 0, "GGA": 1}

    self.xc_type = xc_type
    self.xc_code = xc_code
    self.ao_deriv = xc_type_table.get(xc_type)
    assert self.ao_deriv is not None

    is_hyb_xc = ni.libxc.is_hybrid_xc(kmf.xc)
    assert not is_hyb_xc

    # TODO: fix it
    kpts = kmf.kpts
    kmesh = [1, 1, 3]

    self.smf = k2gamma(kmf, kmesh)
    self.scell = self.smf.cell.copy(deep=True)
    self.scell.verbose = self.verbose
    self.scell.build()

    self.smf.conv_tol = kmf.conv_tol
    self.smf._numint = MultiGridNumInt2(self.scell)
    self.smf.verbose = self.verbose

    def kernel(self, disp=None):
        log = logger.new_logger(self, self.verbose)

        # sanity check
        assert isinstance(self.kmf.with_df, FFTDF)
        assert isinstance(self.smf.with_df, FFTDF)

        log.info("Running SCF for supercell")
        self.smf.kernel()

        dm0 = self.smf.make_rdm1()
        nkpt = len(self.kmf.kpts)
        e_tot_pcell = self.kmf.e_tot * nkpt
        e_tot_scell = self.smf.e_tot
        error = abs(e_tot_pcell - e_tot_scell) / nkpt

        conv_tol = self.smf.conv_tol * nkpt
        if error > conv_tol:
            message = "Energy mismatch: % 6.4e > % 6.4e" % (error, conv_tol)
            message += "\nkmf use %s" % self.kmf._numint.__class__.__name__
            message += "\nsmf use %s" % self.smf._numint.__class__.__name__
            message += "\ne_tot_pcell = %12.8f" % e_tot_pcell
            message += "\ne_tot_scell = %12.8f" % e_tot_scell
            message += "\nPlease increase ke_cutoff"
            log.warn(message)

        scell = self.scell.copy(deep=True)
        nao_scell = scell.nao_nr()
        nao_pcell = self.pcell.nao_nr()
        natm_scell = scell.natm
        natm_pcell = self.pcell.natm

        ni = NumInt()
        assert isinstance(ni, NumInt)

        hermi = 1
        xc_code = self.xc_code
        xc_type = self.xc_type
        ao_deriv = self.ao_deriv

        mesh = scell.mesh
        coul_g = tools.get_coulG(scell, mesh=mesh)
        grid = self.smf.grids
        gx = grid.coords
        gw = grid.weights
        ng = gx.shape[0]

        phi0 = ni.eval_ao(scell, gx, deriv=ao_deriv)

        mask = None
        shls_slice = (0, self.scell.nbas)
        ao_loc = self.scell.ao_loc_nr()

        def wv_phi(f, w):
            mask = None
            f = f.real.copy()
            w = w.real.copy()

            v = ni._vxc_mat(
                self.scell, f, w, mask, xc_type, shls_slice, ao_loc, 1
            )

            v = v + v.conj().swapaxes(-2, -1)
            return v.real

        def wv_and_grad(iatm, x, sign, d=1e-3):
            x0 = scell.atom_coords()
            dx = numpy.zeros_like(x0)
            dx[iatm][x] = sign * d

            c = scell.copy(deep=True)
            c.set_geom_(x0 + dx, inplace=True)
            c.build()

            m = self.smf
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
            rho = gen_rho(0, phi, mask, xc_type)
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

        disp = self.disp if disp is None else disp
        disp_full_supercell = self.disp_full_supercell
        nx = 3 * (natm_scell if disp_full_supercell else natm_pcell)
        for ix in range(nx):
            a, x = divmod(ix, 3)

            e_p, g_p, w_p = wv_and_grad(a, x, +1, disp)
            e_m, g_m, w_m = wv_and_grad(a, x, -1, disp)

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

        assert not disp_full_supercell
        dv = dv.reshape(natm_pcell * 3, nao_scell, nao_scell)
        dg = dg.reshape(natm_pcell * 3, natm_scell * 3)
        self.dv = dv
        self.dg = dg
        return dv, dg


def calculate_electron_phonon_coupling(mf_kc=None, mf_qc=None, disp=1e-4):
    qc = mf_qc.kpts
    kc = mf_kc.kpts

    pcell = mf_kc.cell.copy()
    pcell.verbose = 10

    natm = pcell.natm
    nx = natm * 3
    nao = pcell.nao_nr()
    nk = len(kc)
    nq = len(qc)

    assert mf_qc.converged
    assert mf_kc.converged
    dv, dg = calculate_dv_dg(mf_qc, disp=1e-4)

    dv = dv.reshape(natm * 3, nq, nao, nq, nao)
    dg = dg.reshape(natm * 3, natm * nq * 3)

    import sys, os

    sys.path.append("../gwpt-main/")
    from gwpt.tools.gamma2k import gamma2k_hess

    pcell.verbose = 10
    mass = pcell.atom_mass_list() * MP_ME
    sqrt_mm = numpy.sqrt(mass[:, None] * mass[None, :])

    h_q = dg_g2k(dg, pcell=pcell, qc=qc, kc=kc)
    g_qk_xmn = dv_g2k(dv, pcell=pcell, qc=qc, kc=kc)

    res = eigh_hess(h_q / sqrt_mm[None, :, None, :, None])
    freq_mode_q = []
    coeff_mode_q = []
    for fq, cq in zip(res[0], res[1], strict=True):
        sqrt_mf = numpy.sqrt(2 * mass[:, None] * fq)
        sqrt_mf = sqrt_mf.reshape(natm, -1)
        cq = cq.reshape(natm, 3, -1)

        freq_mode_q.append(fq)
        coeff_mode_q.append(cq / sqrt_mf[:, None, :])
    freq_mode_q = numpy.array(freq_mode_q).reshape(nq, -1)
    coeff_mode_q = numpy.array(coeff_mode_q).reshape(nq, nx, -1)
    num_mode = coeff_mode_q.shape[-1]

    coeff_band_k = numpy.array(kmf.mo_coeff).reshape(nk, nao, -1)
    num_band = coeff_band_k.shape[-1]

    kconserv2 = get_kconserv2(pcell, kc, qc)

    from gwpt.tools.gamma2k import vmat0_to_vmatq_new

    res = vmat0_to_vmatq_new(dv, coeff_mode_q, pcell, qc, coeff_k=coeff_band_k)
    g_qk_lmn_ref, g_qk_xmn_ref = res
    g_qk_lmn_ref = numpy.array(g_qk_lmn_ref)

    g_qk_lmn = []
    for q in range(nq):
        cq = coeff_mode_q[q]
        fq = freq_mode_q[q]
        for k1 in range(nk):
            ck1 = coeff_band_k[k1]
            k2 = kconserv2[k1, q]
            ck2 = coeff_band_k[k2]

            scripts = "xmn,xl,mp,nq->lpq"
            operand = (g_qk_xmn[q, k1], cq, ck2.conj(), ck1)
            # scripts = "xmn,xl->lmn"
            # operand = (g_qk_xmn[q, k1], cq)
            g_qk_lmn.append(lib.einsum(scripts, *operand))

    g_qk_lmn = numpy.array(g_qk_lmn)
    g_qk_lmn_sol = g_qk_lmn.reshape(nq, nk, num_mode, num_band, num_band)

    err = abs(g_qk_lmn_ref - g_qk_lmn_sol).max()
    print(f"g_qk_lmn error = {err:6.2e}")

    err = abs(g_qk_xmn - g_qk_xmn_ref).max()
    print(f"g_qk_xmn error = {err:6.2e}")
    assert 1 == 2

    # for q in range(nq):
    #     for k in range(nk):
    #         for l in range(num_mode):
    #             print(f"\nq = {q}, k = {k}, l = {l}")
    #             print(f"{g_qk_lmn_ref[q, k, l].shape = }")
    #             numpy.savetxt(
    #                 pcell.stdout,
    #                 g_qk_lmn_ref[q, k, l].real,
    #                 fmt="% 6.2e",
    #                 delimiter=", ",
    #             )
    #             print(f"{g_qk_lmn_sol[q, k, l].shape = }")
    #             numpy.savetxt(
    #                 pcell.stdout,
    #                 g_qk_lmn_sol[q, k, l].real,
    #                 fmt="% 6.2e",
    #                 delimiter=", ",
    #             )
    #             err = abs(g_qk_lmn_ref[q, k, l] - g_qk_lmn_sol[q, k, l]).max()
    #             print(f"g_qk_lmn error = {err:6.2e}")
    #             assert err < 1e-8


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

    mf = pbc.dft.KRKS(cell, qc)
    mf.xc = "LDA"
    mf.conv_tol = 1e-6
    mf.conv_tol_grad = 1e-4
    mf.verbose = 4
    mf.kernel()

    elph = ElectronPhononCoupling(mf)
    dvdg_file = "dvdg.h5"

    import os, h5py

    if os.path.exists(dvdg_file):
        with h5py.File(dvdg_file, "r") as f:
            dv = f["dv"][:]
            dg = f["dg"][:]
    else:
        dv, dg = elph.kernel(disp=1e-4)
        with h5py.File(dvdg_file, "w") as f:
            f.create_dataset("dv", data=dv)
            f.create_dataset("dg", data=dg)

    assert dv.shape == (natm * 3, nao * nqc, nao * nqc)
    assert dg.shape == (natm * 3, natm * nqc * 3)

    # kc_mesh = [6, 6, 6]
    kc_mesh = qc_mesh
    kc = cell.make_kpts(kc_mesh)
    nkc = len(kc)
    # any kc + qc should be in kc

    kmf = pbc.dft.KRKS(cell, kc)
    kmf.xc = "LDA"
    kmf.conv_tol = 1e-6
    kmf.conv_tol_grad = 1e-4
    kmf.verbose = 4
    kmf.kernel()

    g_qklpq = gamma2k(elph, kmf, dv, dg)
    print(f"{g_qklpq.shape = }")

    # kf_mesh = [12, 12, 12]
    # qf_mesh = [12, 12, 12]
    # g_qklpq_f = None  # (nqc, nkf, nkc, nmode, nmo, nmo)
