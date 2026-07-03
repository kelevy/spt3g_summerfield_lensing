# Run all validation test
import matplotlib.patches as patches
import scipy.stats as stats
import numpy as np

def ctext(s,qe,e):
    if e: print("\033[31m%s %-3s -- pass \033[0m"%(s,qe))
    else: print("\033[31m%s %-3s -- fail \033[0m"%(s,qe))

def parse_path(config):

    psname = config["pspec"]["psname"]

    dir_base = config["lensrec"]["dir_out"].format(
            rectype=config["lensrec"]["rectype"],
            runname=config["base"]["runname"],
            lmaxT=config["lensrec"]["lmaxT"],
            lminT=config["lensrec"]["lminT"],
            lminP=config["lensrec"]["lminP"],
            lmaxP=config["lensrec"]["lmaxP"],
            mmin=config["lensrec"]["mmin"],
        )

    if psname is not None and psname != "":
        dir_cls = dir_base + f"/clkk_polspice_{psname}_nops/"
    else:
        dir_cls = dir_base + f"/clkk_polspice_nops/"

    return dir_cls

def test_simmean(dict,qe):
    #x = fid.format(qe=qe)
    ctext('sim mean',qe, np.all(dict[qe].item()['ratio']['err']*0.3 < dict[qe].item()['ratio']['rcl']) )


def test_unl(dict,qe,nsims_unl=50, nsims_cov=500):

    cls = dict[qe].item()['unl']['rdl']
    arr = dict[qe].item()['unl']['arr']
    
    # hartlap based on number of sims used to compute error
    hartlap = (nsims_cov - len(cls) - 2)/(nsims_cov - 1)

    # dividing by the number of realizations averaging over
    cov = np.cov(arr)/nsims_unl

    icov = np.linalg.pinv(cov)*hartlap
    chi2 = cls@icov@cls
    pte = stats.chi2.sf(chi2, len(dict[qe].item()['unl']['rl']))

    if pte>0.05 and pte<0.95:
        print(f"\033[32munlensed {qe:<3s} -- pass (pte={pte:.4f}) \033[0m")
    else:
        print(f"\033[31munlensed {qe:<3s} -- fail (pte={pte:.4f}) \033[0m")
        

def test_curl(dict, qe, nsims=500):
    cls = dict[qe].item()['curl']['rdl']
    arr = dict[qe].item()['curl']['arr']
    
    hartlap = (nsims - len(cls) - 2)/(nsims - 1)
    
    cov = np.cov(arr)

    icov = np.linalg.pinv(cov)*hartlap
    chi2 = cls@icov@cls
    pte = stats.chi2.sf(chi2, len(dict[qe].item()['curl']['rl']))
    
    if pte>0.05 and pte<0.95:
        print(f"\033[32mcurl {qe:<3s} -- pass (pte={pte:.4f}) \033[0m")
    else:
        print(f"\033[31mcurl {qe:<3s} -- fail (pte={pte:.4f}) \033[0m")


def diff_test(dict,qe1,dtype1,cltype1,qe2,dtype2,cltype2):
    
    rl1 = dict[qe1].item()[dtype1]['rl']
    rl2 = dict[qe2].item()[dtype2]['rl']
    assert np.allclose(rl1, rl2), 'mismatch in rl1 vs rl2'

    darr1 = dict[qe1].item()[dtype1]['arr']
    darr2 = dict[qe2].item()[dtype2]['arr']
    nsims = darr1.shape[1]

    dvec1 = dict[qe1].item()[dtype1][cltype1]
    dvec2 = dict[qe2].item()[dtype2][cltype2]
    
    dvec  = dvec1-dvec2
    
    cov   = np.cov(darr1-darr2)

    hartlap = (nsims - len(dvec)/2 - 2) / (nsims - 1)
    icov = np.linalg.pinv(cov)*hartlap
    chi2 = dvec@icov@dvec
    
    pte = stats.chi2.sf(chi2, len(dvec))

    if pte>0.05 and pte<0.95:
        print(f"\033[32mcurl {qe:<3s} -- pass (pte={pte:.4f}) \033[0m")
    else:
        print(f"\033[31mcurl {qe:<3s} -- fail (pte={pte:.4f}) \033[0m")


# TT - PP

# (MV-TT) - PP

# lmax 3000 vs 3500

# lmax 4000 vs 3500

# notch

# Agora


if __name__ == "__main__":

    yaml3000 = '../yaml/sqe/config_sqe_021125_crosstf_lmaxt3500_lmaxp3500_mmin100_fullmask_optimzedinp_v3_withemmtf_highninv_1dcinv_binmaskcinv_v2_mfxxyy.yaml'
    yaml3000 = '../yaml/sqe/config_sqe_021125_crosstf_lmaxt3000_lmaxp3000_mmin100_fullmask_optimzedinp_v3_withemmtf_highninv_1dcinv_binmaskcinv_v2_mfxxyy.yaml'
    yaml4000 = '../yaml/sqe/config_sqe_021125_crosstf_lmaxt4000_lmaxp4000_mmin100_fullmask_optimzedinp_v3_withemmtf_highninv_1dcinv_binmaskcinv_v2_mfxxyy.yaml'

    config = hutils.load_yaml(args.file_yaml)['pspec']['dir_cls']    
    dir_cls = parse_path(config)
    
    dict_fid = np.load(dir_cls + f'cls_summary.npz', allow_pickle=True)

    print('------ Sim mean -------')
    test_simmean(dict_fid, 'tt')
    test_simmean(dict_fid, 'qpp')
    test_simmean(dict_fid, 'qmv')

    print('------ Unlensed -------')
    test_unl(dict_fid, 'tt')
    test_unl(dict_fid, 'qpp')
    test_unl(dict_fid, 'qmv')

    print('-------- Curl ---------')
    test_curl(dict_fid, 'tt')
    test_curl(dict_fid, 'qpp')
    test_curl(dict_fid, 'qmv')

    print('------3500 vs 3000 ---------')
    dir_cls = parse_path(hutils.load_yaml(yaml3000)['pspec']['dir_cls'])
    dict_3000 = np.load(dir_cls+f'cls_summary.npz', allow_pickle=True)

    diff_test(dict_fid, 'tt', 'fid3500', 'rdl', dict_3000, 'tt', 'fid3000', 'rdl')
    diff_test(dict_fid, 'qpp', 'fid3500', 'rdl', dict_3000, 'qpp', 'fid3000', 'rdl')
    diff_test(dict_fid, 'qmv', 'fid3500', 'rdl', dict_3000, 'qmv', 'fid3000', 'rdl')

    print('------3500 vs 4000 ---------')
    dir_cls = parse_path(hutils.load_yaml(yaml4000)['pspec']['dir_cls'])
    dict_4000 = np.load(dir_cls+f'cls_summary.npz', allow_pickle=True)

    diff_test(dict_fid, 'tt', 'fid3500', 'rdl', 'tt', dict_4000, 'fid', 'rdl')
    diff_test(dict_fid, 'qpp', 'fid3500', 'rdl', 'qpp', dict_4000, 'fid', 'rdl')
    diff_test(dict_fid, 'qmv', 'fid3500', 'rdl', 'qmv', dict_4000, 'fid', 'rdl')

    print('---------- notch ---------')
    dir_cls = parse_path(hutils.load_yaml(yamlnotch)['pspec']['dir_cls'])
    dict_notch = np.load(dir_cls+f'cls_summary.npz', allow_pickle=True)

    diff_test(dict_fid, 'tt', 'fid3500', 'rdl', 'tt', dict_notch, 'fid', 'rdl')
    diff_test(dict_fid, 'qpp', 'fid3500', 'rdl', 'qpp', dict_notch, 'fid', 'rdl')
    diff_test(dict_fid, 'qmv', 'fid3500', 'rdl', 'qmv', dict_notch, 'fid', 'rdl')
