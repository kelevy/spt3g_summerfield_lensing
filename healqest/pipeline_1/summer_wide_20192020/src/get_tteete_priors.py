import numpy as np

# Define the calibration numbers
# pasted from https://sptlocal.grid.uchicago.edu/~yomori/20192020_lensing/Tcal/v3/spt3g20192020_tcal.html 
# after "The Tcal values for using internal calibration for 90 and 220 GHz and external for 150 GHz are:"
# UPDATED UNCERTAINTIES ONLY to https://southpoletelescope.slack.com/archives/D034YCRJM1U/p1708560154573929
tcal_subfields = {
    '44.75': {'090GHz' :1.058,
              '150GHz' :1.013,
              '220GHz' :0.986},
    '52.25': {'090GHz' :1.068,
              '150GHz' :1.033,
              '220GHz' :0.999},
    '59.75': {'090GHz' :1.071,
              '150GHz' :1.001,
              '220GHz' :1.002},
    '67.25': {'090GHz' :1.081,
              '150GHz' :1.014,
              '220GHz' :1.007},
}
tcal_uncertainties_subfields = {
    '44.75': {'090GHz' :0.002,
              '150GHz' :0.004,
              '220GHz' :0.008},
    '52.25': {'090GHz' :0.001,
              '150GHz' :0.003,
              '220GHz' :0.006},
    '59.75': {'090GHz' :0.002,
              '150GHz' :0.004,
              '220GHz' :0.006},
    '67.25': {'090GHz' :0.002,
              '150GHz' :0.008,
              '220GHz' :0.006},
}
# pasted from https://pole.uchicago.edu/spt3g/images/20231009_EETETT_Updates.pdf
qcal_subfields = {
    '44.75': {'090GHz' :1.00843208,
              '150GHz' :1.10564022,
              '220GHz' :0.89661709},
    '52.25': {'090GHz' :0.99354970,
              '150GHz' :1.03056826,
              '220GHz' :0.93201403},
    '59.75': {'090GHz' :0.99965858,
              '150GHz' :1.06459057,
              '220GHz' :0.92936020},
    '67.25': {'090GHz' :0.9991642,
              '150GHz' :1.08950970,
              '220GHz' :0.90723602},
}
qcal_uncertainties_subfields = {
    '44.75': {'090GHz' :0.00188439,
              '150GHz' :0.03121099,
              '220GHz' :0.00694877},
    '52.25': {'090GHz' :0.00181015,
              '150GHz' :0.01703644,
              '220GHz' :0.00726865},
    '59.75': {'090GHz' :0.00162354,
              '150GHz' :0.03355104,
              '220GHz' :0.00601437},
    '67.25': {'090GHz' :0.0027536,
              '150GHz' :0.02898794,
              '220GHz' :0.00940786},
}
ucal_subfields = {
    '44.75': {'090GHz' :1.02387141,
              '150GHz' :1.07676874,
              '220GHz' :0.91167875},
    '52.25': {'090GHz' :1.02150085,
              '150GHz' :1.11613421,
              '220GHz' :0.91327965},
    '59.75': {'090GHz' :1.01292663,
              '150GHz' :1.12797701,
              '220GHz' :0.89524785},
    '67.25': {'090GHz' :1.01579352,
              '150GHz' :1.10619347,
              '220GHz' :0.90321213},
}
ucal_uncertainties_subfields = {
    '44.75': {'090GHz' :0.00239291,
              '150GHz' :0.01662462,
              '220GHz' :0.00438890},
    '52.25': {'090GHz' :0.00163750,
              '150GHz' :0.02017224,
              '220GHz' :0.00596226},
    '59.75': {'090GHz' :0.00211072,
              '150GHz' :0.02632582,
              '220GHz' :0.00491886},
    '67.25': {'090GHz' :0.00230582,
              '150GHz' :0.02579781,
              '220GHz' :0.00627514},
}

# planck uncertainty
planck_uncertainties = {
    'tcal' :0.0025,
    'pcal' :0.00509,
}

# compute full field uncertainties from subfields
# add planck only for 150GHz
tcal_uncertainties = {}
qcal_uncertainties = {}
ucal_uncertainties = {}
pcal_uncertainties = {}
print('----------------------------------------------------------')
for band in ['090GHz', '150GHz', '220GHz']:
    tcal_uncertainties[band] = 1/np.sqrt(np.sum([1/tcal_uncertainties_subfields[subfield][band]**2 for subfield in tcal_subfields.keys()]))
    # tcal_uncertainties[band] = np.sqrt(np.sum([tcal_uncertainties_subfields[subfield][band]**2 for subfield in tcal_subfields.keys()]))
    qcal_uncertainties[band] = 1/np.sqrt(np.sum([1/qcal_uncertainties_subfields[subfield][band]**2 for subfield in qcal_subfields.keys()]))
    ucal_uncertainties[band] = 1/np.sqrt(np.sum([1/ucal_uncertainties_subfields[subfield][band]**2 for subfield in ucal_subfields.keys()]))
    print(f'{band} tcal uncertainty / variance: {tcal_uncertainties[band]:.3e}, {tcal_uncertainties[band]**2:.3e}')
    print(f'{band} qcal uncertainty / variance: {qcal_uncertainties[band]:.3e}, {qcal_uncertainties[band]**2:.3e}')
    print(f'{band} ucal uncertainty / variance: {ucal_uncertainties[band]:.3e}, {ucal_uncertainties[band]**2:.3e}')
    pcal_uncertainties[band] = 1/np.sqrt(1/qcal_uncertainties[band]**2 + 1/ucal_uncertainties[band]**2)
    if band == '150GHz':
        tcal_uncertainties[band] = np.sqrt(tcal_uncertainties[band]**2 + planck_uncertainties["tcal"]**2)
        pcal_uncertainties[band] = np.sqrt(pcal_uncertainties[band]**2 + planck_uncertainties["pcal"]**2)
        #print(pcal_uncertainties[band],planck_uncertainties["pcal"])
    print(f'{band} pcal uncertainty / variance: {pcal_uncertainties[band]:.3e}, {pcal_uncertainties[band]**2:.3e}')
    print('----------------------------------------------------------')

