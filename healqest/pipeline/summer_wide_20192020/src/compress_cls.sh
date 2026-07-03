#!/bin/bash

#dir=/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/lensrec/sqe021125/sqe/crosstf_v5_lmint500_lminp500_lmaxt3500_lmaxp3500_mmin100_crosstf_ananinv_v3_withemmtf_highninv_1dcinv_binmaskcinv_v2a//clkk_polspice_mfxxyy_nops/
#dir=/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/lensrec/sqe021125/sqe/crosstf_v5_lmint500_lminp500_lmaxt3500_lmaxp3500_mmin100_crosstf_ananinv_v3_withemmtf_highninv_1dcinv_binmaskcinv_v2a_notch//clkk_polspice_nops/
dir=/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/lensrec/sqe021125/sqe/crosstf_v5_lmint500_lminp500_lmaxt3000_lmaxp3000_mmin100_crosstf_ananinv_v3_withemmtf_highninv_1dcinv_binmaskcinv_v2a//clkk_polspice_mfxxyy_nops/
dir=/lcrc/project/SPT3G/users/ac.yomori/projects/spt3g_lensing_20192020/lensrec/sqe021125/sqe/crosstf_v5_lmint500_lminp500_lmaxt4000_lmaxp4000_mmin100_crosstf_ananinv_v3_withemmtf_highninv_1dcinv_binmaskcinv_v2a//clkk_polspice_nops/

qe=qmv
spec=kk

python compress_cls.py $dir ${qe} ${spec} dddd
python compress_cls.py $dir ${qe} ${spec} xxxx
python compress_cls.py $dir ${qe} ${spec} xyxy
python compress_cls.py $dir ${qe} ${spec} xyyx
python compress_cls.py $dir ${qe} ${spec} xdxd
python compress_cls.py $dir ${qe} ${spec} xddx
python compress_cls.py $dir ${qe} ${spec} dxdx
python compress_cls.py $dir ${qe} ${spec} dxxd
python compress_cls.py $dir ${qe} ${spec} abab
python compress_cls.py $dir ${qe} ${spec} abba
python compress_cls.py $dir ${qe} ${spec} uuuu

ls $dir | wc -l
