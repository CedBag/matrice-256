# Rapport de benchmarks scientifiques v5.0 : Matrice-256 (FaNet v4.2)

- Date : 2026-11-22 23:26:14
- Plateforme : Linux-6.18.55+-x86_64-with-glibc2.39
- PyTorch : 2.11.0+cu130 | Device : cuda (Tesla T4)
- Durée totale : 6267.0 s
- Configuration : 5-Fold x 5 seeds, 80 époques max (early stopping)

## Exp 1 : Étude d'ablation (F1, ROC-AUC, PR-AUC, Brier, ECE pondérée)

| dataset | variant | f1_mean | f1_std | roc_auc_mean | pr_auc_mean | brier_mean | ece_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pima_diabetes | B0_LogReg | 0.6385 | 0.0374 | 0.8322 | 0.7254 | 0.1568 | 0.0754 |
| pima_diabetes | B1_RF_Reg | 0.6303 | 0.0508 | 0.8303 | 0.7099 | 0.1588 | 0.0813 |
| pima_diabetes | B2_HistGB | 0.6204 | 0.0440 | 0.8145 | 0.6907 | 0.1686 | 0.0950 |
| pima_diabetes | M0_MLP_Iso | 0.6048 | 0.1050 | 0.8208 | 0.7004 | 0.1641 | 0.0845 |
| pima_diabetes | A1_FaNet_NoLoss | 0.6334 | 0.0538 | 0.8303 | 0.7131 | 0.1602 | 0.0808 |
| pima_diabetes | A2_FaNet_Const05 | 0.6382 | 0.0398 | 0.8309 | 0.7144 | 0.1594 | 0.0789 |
| pima_diabetes | A3_FaNet_Permuted | 0.6217 | 0.0595 | 0.8319 | 0.7147 | 0.1597 | 0.0770 |
| pima_diabetes | A4_FaNet_FaDual | 0.6137 | 0.0612 | 0.8303 | 0.7131 | 0.1607 | 0.0798 |
| pima_diabetes | F1_FaNet_Full | 0.6308 | 0.0422 | 0.8318 | 0.7149 | 0.1590 | 0.0774 |
| breast_cancer | B0_LogReg | 0.9683 | 0.0207 | 0.9948 | 0.9936 | 0.0199 | 0.0290 |
| breast_cancer | B1_RF_Reg | 0.9463 | 0.0226 | 0.9913 | 0.9889 | 0.0334 | 0.0499 |
| breast_cancer | B2_HistGB | 0.9510 | 0.0279 | 0.9925 | 0.9903 | 0.0278 | 0.0346 |
| breast_cancer | M0_MLP_Iso | 0.9635 | 0.0200 | 0.9912 | 0.9901 | 0.0235 | 0.0278 |
| breast_cancer | A1_FaNet_NoLoss | 0.9543 | 0.0261 | 0.9955 | 0.9940 | 0.0248 | 0.0307 |
| breast_cancer | A2_FaNet_Const05 | 0.9594 | 0.0250 | 0.9947 | 0.9934 | 0.0240 | 0.0304 |
| breast_cancer | A3_FaNet_Permuted | 0.9641 | 0.0220 | 0.9952 | 0.9940 | 0.0215 | 0.0285 |
| breast_cancer | A4_FaNet_FaDual | 0.9568 | 0.0231 | 0.9941 | 0.9931 | 0.0261 | 0.0339 |
| breast_cancer | F1_FaNet_Full | 0.9605 | 0.0253 | 0.9949 | 0.9935 | 0.0229 | 0.0282 |
| agri_high_water | B0_LogReg | 0.8401 | 0.0288 | 0.9841 | 0.8835 | 0.0440 | 0.0612 |
| agri_high_water | B1_RF_Reg | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0009 | 0.0083 |
| agri_high_water | B2_HistGB | 0.9950 | 0.0066 | 0.9998 | 0.9974 | 0.0011 | 0.0015 |
| agri_high_water | M0_MLP_Iso | 0.9899 | 0.0158 | 0.9999 | 0.9992 | 0.0022 | 0.0028 |
| agri_high_water | A1_FaNet_NoLoss | 0.9908 | 0.0100 | 0.9999 | 0.9994 | 0.0022 | 0.0034 |
| agri_high_water | A2_FaNet_Const05 | 0.9885 | 0.0091 | 0.9999 | 0.9994 | 0.0025 | 0.0034 |
| agri_high_water | A3_FaNet_Permuted | 0.9888 | 0.0085 | 0.9999 | 0.9996 | 0.0025 | 0.0034 |
| agri_high_water | A4_FaNet_FaDual | 0.9850 | 0.0143 | 0.9991 | 0.9898 | 0.0037 | 0.0137 |
| agri_high_water | F1_FaNet_Full | 0.9879 | 0.0100 | 0.9999 | 0.9994 | 0.0027 | 0.0038 |
| agri_rice | B0_LogReg | 0.7299 | 0.0446 | 0.9969 | 0.9494 | 0.0239 | 0.0391 |
| agri_rice | B1_RF_Reg | 0.9543 | 0.0303 | 0.9993 | 0.9844 | 0.0041 | 0.0078 |
| agri_rice | B2_HistGB | 0.9475 | 0.0328 | 0.9996 | 0.9895 | 0.0035 | 0.0048 |
| agri_rice | M0_MLP_Iso | 0.8796 | 0.0597 | 0.9976 | 0.9738 | 0.0090 | 0.0110 |
| agri_rice | A1_FaNet_NoLoss | 0.8409 | 0.0567 | 0.9977 | 0.9573 | 0.0126 | 0.0174 |
| agri_rice | A2_FaNet_Const05 | 0.8454 | 0.0523 | 0.9977 | 0.9580 | 0.0122 | 0.0166 |
| agri_rice | A3_FaNet_Permuted | 0.8431 | 0.0554 | 0.9978 | 0.9591 | 0.0118 | 0.0159 |
| agri_rice | A4_FaNet_FaDual | 0.8380 | 0.0589 | 0.9974 | 0.9540 | 0.0129 | 0.0271 |
| agri_rice | F1_FaNet_Full | 0.8341 | 0.0661 | 0.9973 | 0.9539 | 0.0130 | 0.0171 |

## Exp 1b : Tests appariés de Wilcoxon (F1_FaNet_Full vs contrôles)

| dataset | comparison | metric | mean_diff | wilcoxon_p | significant_005 |
| --- | --- | --- | --- | --- | --- |
| pima_diabetes | F1_FaNet_Full vs B0_LogReg | f1 | -0.0077 | 0.4470 | False |
| pima_diabetes | F1_FaNet_Full vs B0_LogReg | roc_auc | -0.0004 | 0.9571 | False |
| pima_diabetes | F1_FaNet_Full vs B0_LogReg | pr_auc | -0.0105 | 0.0296 | True |
| pima_diabetes | F1_FaNet_Full vs B0_LogReg | brier | 0.0022 | 0.1336 | False |
| pima_diabetes | F1_FaNet_Full vs B0_LogReg | ece | 0.0020 | 0.4742 | False |
| pima_diabetes | F1_FaNet_Full vs B1_RF_Reg | f1 | 0.0005 | 0.9158 | False |
| pima_diabetes | F1_FaNet_Full vs B1_RF_Reg | roc_auc | 0.0015 | 0.5424 | False |
| pima_diabetes | F1_FaNet_Full vs B1_RF_Reg | pr_auc | 0.0050 | 0.3525 | False |
| pima_diabetes | F1_FaNet_Full vs B1_RF_Reg | brier | 0.0002 | 0.9789 | False |
| pima_diabetes | F1_FaNet_Full vs B1_RF_Reg | ece | -0.0038 | 0.4261 | False |
| pima_diabetes | F1_FaNet_Full vs B2_HistGB | f1 | 0.0104 | 0.3957 | False |
| pima_diabetes | F1_FaNet_Full vs B2_HistGB | roc_auc | 0.0174 | 0.0001 | True |
| pima_diabetes | F1_FaNet_Full vs B2_HistGB | pr_auc | 0.0242 | 0.0046 | True |
| pima_diabetes | F1_FaNet_Full vs B2_HistGB | brier | -0.0096 | 0.0001 | True |
| pima_diabetes | F1_FaNet_Full vs B2_HistGB | ece | -0.0176 | 0.0042 | True |
| pima_diabetes | F1_FaNet_Full vs M0_MLP_Iso | f1 | 0.0260 | 0.2411 | False |
| pima_diabetes | F1_FaNet_Full vs M0_MLP_Iso | roc_auc | 0.0111 | 0.0045 | True |
| pima_diabetes | F1_FaNet_Full vs M0_MLP_Iso | pr_auc | 0.0146 | 0.0088 | True |
| pima_diabetes | F1_FaNet_Full vs M0_MLP_Iso | brier | -0.0051 | 0.0061 | True |
| pima_diabetes | F1_FaNet_Full vs M0_MLP_Iso | ece | -0.0071 | 0.3123 | False |
| pima_diabetes | F1_FaNet_Full vs A1_FaNet_NoLoss | f1 | -0.0025 | 0.2627 | False |
| pima_diabetes | F1_FaNet_Full vs A1_FaNet_NoLoss | roc_auc | 0.0016 | 0.0851 | False |
| pima_diabetes | F1_FaNet_Full vs A1_FaNet_NoLoss | pr_auc | 0.0018 | 0.5249 | False |
| pima_diabetes | F1_FaNet_Full vs A1_FaNet_NoLoss | brier | -0.0012 | 0.2099 | False |
| pima_diabetes | F1_FaNet_Full vs A1_FaNet_NoLoss | ece | -0.0034 | 0.5249 | False |
| pima_diabetes | F1_FaNet_Full vs A2_FaNet_Const05 | f1 | -0.0074 | 0.3720 | False |
| pima_diabetes | F1_FaNet_Full vs A2_FaNet_Const05 | roc_auc | 0.0009 | 0.3002 | False |
| pima_diabetes | F1_FaNet_Full vs A2_FaNet_Const05 | pr_auc | 0.0006 | 0.6721 | False |
| pima_diabetes | F1_FaNet_Full vs A2_FaNet_Const05 | brier | -0.0004 | 0.5602 | False |
| pima_diabetes | F1_FaNet_Full vs A2_FaNet_Const05 | ece | -0.0015 | 0.7712 | False |
| pima_diabetes | F1_FaNet_Full vs A3_FaNet_Permuted | f1 | 0.0091 | 0.5901 | False |
| pima_diabetes | F1_FaNet_Full vs A3_FaNet_Permuted | roc_auc | -0.0001 | 0.7915 | False |
| pima_diabetes | F1_FaNet_Full vs A3_FaNet_Permuted | pr_auc | 0.0002 | 1.0000 | False |
| pima_diabetes | F1_FaNet_Full vs A3_FaNet_Permuted | brier | -0.0007 | 0.8325 | False |
| pima_diabetes | F1_FaNet_Full vs A3_FaNet_Permuted | ece | 0.0005 | 0.6721 | False |
| pima_diabetes | F1_FaNet_Full vs A4_FaNet_FaDual | f1 | 0.0171 | 0.0335 | True |
| pima_diabetes | F1_FaNet_Full vs A4_FaNet_FaDual | roc_auc | 0.0016 | 0.0274 | True |
| pima_diabetes | F1_FaNet_Full vs A4_FaNet_FaDual | pr_auc | 0.0018 | 0.2099 | False |
| pima_diabetes | F1_FaNet_Full vs A4_FaNet_FaDual | brier | -0.0017 | 0.0393 | True |
| pima_diabetes | F1_FaNet_Full vs A4_FaNet_FaDual | ece | -0.0024 | 0.3254 | False |
| breast_cancer | F1_FaNet_Full vs B0_LogReg | f1 | -0.0078 | 0.0401 | True |
| breast_cancer | F1_FaNet_Full vs B0_LogReg | roc_auc | 0.0001 | 0.8484 | False |
| breast_cancer | F1_FaNet_Full vs B0_LogReg | pr_auc | -0.0001 | 0.6682 | False |
| breast_cancer | F1_FaNet_Full vs B0_LogReg | brier | 0.0030 | 0.1014 | False |
| breast_cancer | F1_FaNet_Full vs B0_LogReg | ece | -0.0008 | 0.4261 | False |
| breast_cancer | F1_FaNet_Full vs B1_RF_Reg | f1 | 0.0142 | 0.0173 | True |
| breast_cancer | F1_FaNet_Full vs B1_RF_Reg | roc_auc | 0.0036 | 0.0160 | True |
| breast_cancer | F1_FaNet_Full vs B1_RF_Reg | pr_auc | 0.0047 | 0.0020 | True |
| breast_cancer | F1_FaNet_Full vs B1_RF_Reg | brier | -0.0105 | 0.0003 | True |
| breast_cancer | F1_FaNet_Full vs B1_RF_Reg | ece | -0.0216 | 0.0000 | True |
| breast_cancer | F1_FaNet_Full vs B2_HistGB | f1 | 0.0095 | 0.0386 | True |
| breast_cancer | F1_FaNet_Full vs B2_HistGB | roc_auc | 0.0024 | 0.0101 | True |
| breast_cancer | F1_FaNet_Full vs B2_HistGB | pr_auc | 0.0033 | 0.0056 | True |
| breast_cancer | F1_FaNet_Full vs B2_HistGB | brier | -0.0049 | 0.0451 | True |
| breast_cancer | F1_FaNet_Full vs B2_HistGB | ece | -0.0064 | 0.0219 | True |
| breast_cancer | F1_FaNet_Full vs M0_MLP_Iso | f1 | -0.0031 | 0.4651 | False |
| breast_cancer | F1_FaNet_Full vs M0_MLP_Iso | roc_auc | 0.0038 | 0.0037 | True |
| breast_cancer | F1_FaNet_Full vs M0_MLP_Iso | pr_auc | 0.0034 | 0.0046 | True |
| breast_cancer | F1_FaNet_Full vs M0_MLP_Iso | brier | -0.0006 | 0.8532 | False |
| breast_cancer | F1_FaNet_Full vs M0_MLP_Iso | ece | 0.0004 | 0.9368 | False |
| breast_cancer | F1_FaNet_Full vs A1_FaNet_NoLoss | f1 | 0.0061 | 0.0613 | False |
| breast_cancer | F1_FaNet_Full vs A1_FaNet_NoLoss | roc_auc | -0.0006 | 0.1936 | False |
| breast_cancer | F1_FaNet_Full vs A1_FaNet_NoLoss | pr_auc | -0.0005 | 0.2473 | False |
| breast_cancer | F1_FaNet_Full vs A1_FaNet_NoLoss | brier | -0.0018 | 0.4578 | False |
| breast_cancer | F1_FaNet_Full vs A1_FaNet_NoLoss | ece | -0.0025 | 0.3123 | False |
| breast_cancer | F1_FaNet_Full vs A2_FaNet_Const05 | f1 | 0.0011 | 0.9679 | False |
| breast_cancer | F1_FaNet_Full vs A2_FaNet_Const05 | roc_auc | 0.0002 | 0.8483 | False |
| breast_cancer | F1_FaNet_Full vs A2_FaNet_Const05 | pr_auc | 0.0001 | 0.8076 | False |
| breast_cancer | F1_FaNet_Full vs A2_FaNet_Const05 | brier | -0.0010 | 0.9368 | False |
| breast_cancer | F1_FaNet_Full vs A2_FaNet_Const05 | ece | -0.0021 | 0.4261 | False |
| breast_cancer | F1_FaNet_Full vs A3_FaNet_Permuted | f1 | -0.0036 | 0.4859 | False |
| breast_cancer | F1_FaNet_Full vs A3_FaNet_Permuted | roc_auc | -0.0003 | 0.2839 | False |
| breast_cancer | F1_FaNet_Full vs A3_FaNet_Permuted | pr_auc | -0.0004 | 0.1808 | False |
| breast_cancer | F1_FaNet_Full vs A3_FaNet_Permuted | brier | 0.0014 | 0.4578 | False |
| breast_cancer | F1_FaNet_Full vs A3_FaNet_Permuted | ece | -0.0003 | 0.5424 | False |
| breast_cancer | F1_FaNet_Full vs A4_FaNet_FaDual | f1 | 0.0037 | 0.3634 | False |
| breast_cancer | F1_FaNet_Full vs A4_FaNet_FaDual | roc_auc | 0.0009 | 0.9108 | False |
| breast_cancer | F1_FaNet_Full vs A4_FaNet_FaDual | pr_auc | 0.0005 | 0.6612 | False |
| breast_cancer | F1_FaNet_Full vs A4_FaNet_FaDual | brier | -0.0031 | 0.2304 | False |
| breast_cancer | F1_FaNet_Full vs A4_FaNet_FaDual | ece | -0.0057 | 0.0802 | False |
| agri_high_water | F1_FaNet_Full vs B0_LogReg | f1 | 0.1477 | 0.0000 | True |
| agri_high_water | F1_FaNet_Full vs B0_LogReg | roc_auc | 0.0157 | 0.0000 | True |
| agri_high_water | F1_FaNet_Full vs B0_LogReg | pr_auc | 0.1158 | 0.0000 | True |
| agri_high_water | F1_FaNet_Full vs B0_LogReg | brier | -0.0413 | 0.0000 | True |
| agri_high_water | F1_FaNet_Full vs B0_LogReg | ece | -0.0574 | 0.0000 | True |
| agri_high_water | F1_FaNet_Full vs B1_RF_Reg | f1 | -0.0121 | 0.0001 | True |
| agri_high_water | F1_FaNet_Full vs B1_RF_Reg | roc_auc | -0.0001 | 0.0022 | True |
| agri_high_water | F1_FaNet_Full vs B1_RF_Reg | pr_auc | -0.0006 | 0.0022 | True |
| agri_high_water | F1_FaNet_Full vs B1_RF_Reg | brier | 0.0019 | 0.0000 | True |
| agri_high_water | F1_FaNet_Full vs B1_RF_Reg | ece | -0.0045 | 0.0000 | True |
| agri_high_water | F1_FaNet_Full vs B2_HistGB | f1 | -0.0071 | 0.0101 | True |
| agri_high_water | F1_FaNet_Full vs B2_HistGB | roc_auc | 0.0001 | 0.6747 | False |
| agri_high_water | F1_FaNet_Full vs B2_HistGB | pr_auc | 0.0020 | 0.5014 | False |
| agri_high_water | F1_FaNet_Full vs B2_HistGB | brier | 0.0016 | 0.0018 | True |
| agri_high_water | F1_FaNet_Full vs B2_HistGB | ece | 0.0022 | 0.0012 | True |
| agri_high_water | F1_FaNet_Full vs M0_MLP_Iso | f1 | -0.0021 | 0.1239 | False |
| agri_high_water | F1_FaNet_Full vs M0_MLP_Iso | roc_auc | 0.0000 | 0.7832 | False |
| agri_high_water | F1_FaNet_Full vs M0_MLP_Iso | pr_auc | 0.0001 | 0.5505 | False |
| agri_high_water | F1_FaNet_Full vs M0_MLP_Iso | brier | 0.0005 | 0.0255 | True |
| agri_high_water | F1_FaNet_Full vs M0_MLP_Iso | ece | 0.0009 | 0.0105 | True |
| agri_high_water | F1_FaNet_Full vs A1_FaNet_NoLoss | f1 | -0.0030 | 0.1394 | False |
| agri_high_water | F1_FaNet_Full vs A1_FaNet_NoLoss | roc_auc | -0.0000 | 0.2430 | False |
| agri_high_water | F1_FaNet_Full vs A1_FaNet_NoLoss | pr_auc | -0.0001 | 0.2718 | False |
| agri_high_water | F1_FaNet_Full vs A1_FaNet_NoLoss | brier | 0.0005 | 0.2521 | False |
| agri_high_water | F1_FaNet_Full vs A1_FaNet_NoLoss | ece | 0.0004 | 0.3525 | False |
| agri_high_water | F1_FaNet_Full vs A2_FaNet_Const05 | f1 | -0.0006 | 0.5166 | False |
| agri_high_water | F1_FaNet_Full vs A2_FaNet_Const05 | roc_auc | -0.0000 | 0.6490 | False |
| agri_high_water | F1_FaNet_Full vs A2_FaNet_Const05 | pr_auc | -0.0000 | 0.7173 | False |
| agri_high_water | F1_FaNet_Full vs A2_FaNet_Const05 | brier | 0.0003 | 0.5782 | False |
| agri_high_water | F1_FaNet_Full vs A2_FaNet_Const05 | ece | 0.0004 | 0.3666 | False |
| agri_high_water | F1_FaNet_Full vs A3_FaNet_Permuted | f1 | -0.0009 | 0.8645 | False |
| agri_high_water | F1_FaNet_Full vs A3_FaNet_Permuted | roc_auc | -0.0000 | 0.4357 | False |
| agri_high_water | F1_FaNet_Full vs A3_FaNet_Permuted | pr_auc | -0.0002 | 0.6474 | False |
| agri_high_water | F1_FaNet_Full vs A3_FaNet_Permuted | brier | 0.0002 | 0.7310 | False |
| agri_high_water | F1_FaNet_Full vs A3_FaNet_Permuted | ece | 0.0004 | 0.3254 | False |
| agri_high_water | F1_FaNet_Full vs A4_FaNet_FaDual | f1 | 0.0029 | 0.2654 | False |
| agri_high_water | F1_FaNet_Full vs A4_FaNet_FaDual | roc_auc | 0.0008 | 0.0052 | True |
| agri_high_water | F1_FaNet_Full vs A4_FaNet_FaDual | pr_auc | 0.0096 | 0.0045 | True |
| agri_high_water | F1_FaNet_Full vs A4_FaNet_FaDual | brier | -0.0010 | 0.0451 | True |
| agri_high_water | F1_FaNet_Full vs A4_FaNet_FaDual | ece | -0.0100 | 0.0000 | True |
| agri_rice | F1_FaNet_Full vs B0_LogReg | f1 | 0.1042 | 0.0000 | True |
| agri_rice | F1_FaNet_Full vs B0_LogReg | roc_auc | 0.0004 | 0.0381 | True |
| agri_rice | F1_FaNet_Full vs B0_LogReg | pr_auc | 0.0045 | 0.2002 | False |
| agri_rice | F1_FaNet_Full vs B0_LogReg | brier | -0.0109 | 0.0000 | True |
| agri_rice | F1_FaNet_Full vs B0_LogReg | ece | -0.0220 | 0.0000 | True |
| agri_rice | F1_FaNet_Full vs B1_RF_Reg | f1 | -0.1202 | 0.0000 | True |
| agri_rice | F1_FaNet_Full vs B1_RF_Reg | roc_auc | -0.0020 | 0.0000 | True |
| agri_rice | F1_FaNet_Full vs B1_RF_Reg | pr_auc | -0.0304 | 0.0000 | True |
| agri_rice | F1_FaNet_Full vs B1_RF_Reg | brier | 0.0089 | 0.0000 | True |
| agri_rice | F1_FaNet_Full vs B1_RF_Reg | ece | 0.0093 | 0.0000 | True |
| agri_rice | F1_FaNet_Full vs B2_HistGB | f1 | -0.1134 | 0.0000 | True |
| agri_rice | F1_FaNet_Full vs B2_HistGB | roc_auc | -0.0022 | 0.0000 | True |
| agri_rice | F1_FaNet_Full vs B2_HistGB | pr_auc | -0.0356 | 0.0000 | True |
| agri_rice | F1_FaNet_Full vs B2_HistGB | brier | 0.0096 | 0.0000 | True |
| agri_rice | F1_FaNet_Full vs B2_HistGB | ece | 0.0123 | 0.0000 | True |
| agri_rice | F1_FaNet_Full vs M0_MLP_Iso | f1 | -0.0455 | 0.0033 | True |
| agri_rice | F1_FaNet_Full vs M0_MLP_Iso | roc_auc | -0.0002 | 0.0047 | True |
| agri_rice | F1_FaNet_Full vs M0_MLP_Iso | pr_auc | -0.0199 | 0.0000 | True |
| agri_rice | F1_FaNet_Full vs M0_MLP_Iso | brier | 0.0040 | 0.0025 | True |
| agri_rice | F1_FaNet_Full vs M0_MLP_Iso | ece | 0.0061 | 0.0001 | True |
| agri_rice | F1_FaNet_Full vs A1_FaNet_NoLoss | f1 | -0.0068 | 0.2970 | False |
| agri_rice | F1_FaNet_Full vs A1_FaNet_NoLoss | roc_auc | -0.0003 | 0.5358 | False |
| agri_rice | F1_FaNet_Full vs A1_FaNet_NoLoss | pr_auc | -0.0034 | 0.7915 | False |
| agri_rice | F1_FaNet_Full vs A1_FaNet_NoLoss | brier | 0.0004 | 0.5249 | False |
| agri_rice | F1_FaNet_Full vs A1_FaNet_NoLoss | ece | -0.0003 | 0.9578 | False |
| agri_rice | F1_FaNet_Full vs A2_FaNet_Const05 | f1 | -0.0113 | 0.1075 | False |
| agri_rice | F1_FaNet_Full vs A2_FaNet_Const05 | roc_auc | -0.0004 | 0.1410 | False |
| agri_rice | F1_FaNet_Full vs A2_FaNet_Const05 | pr_auc | -0.0040 | 0.2411 | False |
| agri_rice | F1_FaNet_Full vs A2_FaNet_Const05 | brier | 0.0009 | 0.2200 | False |
| agri_rice | F1_FaNet_Full vs A2_FaNet_Const05 | ece | 0.0005 | 0.7712 | False |
| agri_rice | F1_FaNet_Full vs A3_FaNet_Permuted | f1 | -0.0090 | 0.2011 | False |
| agri_rice | F1_FaNet_Full vs A3_FaNet_Permuted | roc_auc | -0.0004 | 0.2176 | False |
| agri_rice | F1_FaNet_Full vs A3_FaNet_Permuted | pr_auc | -0.0052 | 0.3525 | False |
| agri_rice | F1_FaNet_Full vs A3_FaNet_Permuted | brier | 0.0013 | 0.1266 | False |
| agri_rice | F1_FaNet_Full vs A3_FaNet_Permuted | ece | 0.0012 | 0.5424 | False |
| agri_rice | F1_FaNet_Full vs A4_FaNet_FaDual | f1 | -0.0039 | 0.6529 | False |
| agri_rice | F1_FaNet_Full vs A4_FaNet_FaDual | roc_auc | -0.0000 | 0.6478 | False |
| agri_rice | F1_FaNet_Full vs A4_FaNet_FaDual | pr_auc | -0.0000 | 0.9090 | False |
| agri_rice | F1_FaNet_Full vs A4_FaNet_FaDual | brier | 0.0001 | 0.6915 | False |
| agri_rice | F1_FaNet_Full vs A4_FaNet_FaDual | ece | -0.0100 | 0.0000 | True |

## Exp 2 : Robustesse au bruit gaussien

| dataset | noise_sigma | variant | f1_mean | f1_std | pr_auc_mean | pr_auc_std | roc_auc_mean | acc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agri_high_water | 0.0000 | A1_FaNet_NoLoss | 0.9918 | 0.0082 | 0.9992 | 0.0012 | 0.9999 | 0.9977 |
| agri_high_water | 0.0000 | A3_FaNet_Permuted | 0.9837 | 0.0149 | 0.9956 | 0.0089 | 0.9994 | 0.9955 |
| agri_high_water | 0.0000 | B1_RF_Reg | 0.9983 | 0.0038 | 1.0000 | 0.0000 | 1.0000 | 0.9995 |
| agri_high_water | 0.0000 | B2_HistGB | 0.9966 | 0.0076 | 1.0000 | 0.0000 | 1.0000 | 0.9991 |
| agri_high_water | 0.0000 | F1_FaNet_Full | 0.9851 | 0.0108 | 0.9988 | 0.0010 | 0.9998 | 0.9959 |
| agri_high_water | 0.0000 | M0_MLP_Iso | 0.9902 | 0.0106 | 0.9995 | 0.0011 | 0.9999 | 0.9973 |
| agri_high_water | 0.1000 | A1_FaNet_NoLoss | 0.9754 | 0.0109 | 0.9969 | 0.0029 | 0.9995 | 0.9933 |
| agri_high_water | 0.1000 | A3_FaNet_Permuted | 0.9686 | 0.0131 | 0.9929 | 0.0088 | 0.9990 | 0.9914 |
| agri_high_water | 0.1000 | B1_RF_Reg | 0.9419 | 0.0326 | 0.9875 | 0.0082 | 0.9978 | 0.9849 |
| agri_high_water | 0.1000 | B2_HistGB | 0.9023 | 0.0301 | 0.9689 | 0.0212 | 0.9917 | 0.9753 |
| agri_high_water | 0.1000 | F1_FaNet_Full | 0.9693 | 0.0140 | 0.9961 | 0.0031 | 0.9994 | 0.9915 |
| agri_high_water | 0.1000 | M0_MLP_Iso | 0.9821 | 0.0136 | 0.9972 | 0.0050 | 0.9996 | 0.9951 |
| agri_high_water | 0.2000 | A1_FaNet_NoLoss | 0.9426 | 0.0210 | 0.9861 | 0.0093 | 0.9973 | 0.9845 |
| agri_high_water | 0.2000 | A3_FaNet_Permuted | 0.9262 | 0.0208 | 0.9755 | 0.0131 | 0.9958 | 0.9796 |
| agri_high_water | 0.2000 | B1_RF_Reg | 0.8350 | 0.0457 | 0.9318 | 0.0274 | 0.9864 | 0.9593 |
| agri_high_water | 0.2000 | B2_HistGB | 0.8109 | 0.0491 | 0.9223 | 0.0375 | 0.9796 | 0.9540 |
| agri_high_water | 0.2000 | F1_FaNet_Full | 0.9264 | 0.0216 | 0.9793 | 0.0112 | 0.9963 | 0.9799 |
| agri_high_water | 0.2000 | M0_MLP_Iso | 0.9548 | 0.0192 | 0.9901 | 0.0091 | 0.9977 | 0.9877 |
| agri_high_water | 0.3000 | A1_FaNet_NoLoss | 0.8714 | 0.0305 | 0.9450 | 0.0207 | 0.9886 | 0.9665 |
| agri_high_water | 0.3000 | A3_FaNet_Permuted | 0.8504 | 0.0367 | 0.9234 | 0.0270 | 0.9844 | 0.9600 |
| agri_high_water | 0.3000 | B1_RF_Reg | 0.7102 | 0.0543 | 0.8376 | 0.0440 | 0.9615 | 0.9332 |
| agri_high_water | 0.3000 | B2_HistGB | 0.7201 | 0.0534 | 0.8399 | 0.0581 | 0.9440 | 0.9336 |
| agri_high_water | 0.3000 | F1_FaNet_Full | 0.8558 | 0.0378 | 0.9355 | 0.0242 | 0.9869 | 0.9623 |
| agri_high_water | 0.3000 | M0_MLP_Iso | 0.8885 | 0.0264 | 0.9547 | 0.0180 | 0.9858 | 0.9711 |
| agri_high_water | 0.5000 | A1_FaNet_NoLoss | 0.7061 | 0.0396 | 0.8130 | 0.0406 | 0.9545 | 0.9281 |
| agri_high_water | 0.5000 | A3_FaNet_Permuted | 0.7078 | 0.0519 | 0.7875 | 0.0491 | 0.9406 | 0.9249 |
| agri_high_water | 0.5000 | B1_RF_Reg | 0.5403 | 0.0740 | 0.6500 | 0.0749 | 0.9072 | 0.9020 |
| agri_high_water | 0.5000 | B2_HistGB | 0.5792 | 0.0581 | 0.6623 | 0.0546 | 0.8775 | 0.9034 |
| agri_high_water | 0.5000 | F1_FaNet_Full | 0.6943 | 0.0520 | 0.7912 | 0.0527 | 0.9447 | 0.9245 |
| agri_high_water | 0.5000 | M0_MLP_Iso | 0.7297 | 0.0558 | 0.8100 | 0.0533 | 0.9261 | 0.9372 |
| pima_diabetes | 0.0000 | A1_FaNet_NoLoss | 0.6293 | 0.0159 | 0.7148 | 0.0416 | 0.8272 | 0.7539 |
| pima_diabetes | 0.0000 | A3_FaNet_Permuted | 0.6293 | 0.0448 | 0.7055 | 0.0487 | 0.8260 | 0.7617 |
| pima_diabetes | 0.0000 | B1_RF_Reg | 0.6367 | 0.0318 | 0.7001 | 0.0379 | 0.8311 | 0.7682 |
| pima_diabetes | 0.0000 | B2_HistGB | 0.6256 | 0.0294 | 0.6820 | 0.0408 | 0.8160 | 0.7513 |
| pima_diabetes | 0.0000 | F1_FaNet_Full | 0.6155 | 0.0402 | 0.7061 | 0.0393 | 0.8265 | 0.7564 |
| pima_diabetes | 0.0000 | M0_MLP_Iso | 0.6179 | 0.0803 | 0.6908 | 0.0543 | 0.8206 | 0.7512 |
| pima_diabetes | 0.1000 | A1_FaNet_NoLoss | 0.6329 | 0.0234 | 0.7129 | 0.0335 | 0.8262 | 0.7565 |
| pima_diabetes | 0.1000 | A3_FaNet_Permuted | 0.6179 | 0.0517 | 0.7028 | 0.0422 | 0.8247 | 0.7562 |
| pima_diabetes | 0.1000 | B1_RF_Reg | 0.6290 | 0.0365 | 0.6985 | 0.0366 | 0.8309 | 0.7614 |
| pima_diabetes | 0.1000 | B2_HistGB | 0.6237 | 0.0290 | 0.6693 | 0.0399 | 0.8133 | 0.7505 |
| pima_diabetes | 0.1000 | F1_FaNet_Full | 0.6118 | 0.0443 | 0.7075 | 0.0360 | 0.8260 | 0.7536 |
| pima_diabetes | 0.1000 | M0_MLP_Iso | 0.6146 | 0.0705 | 0.6866 | 0.0502 | 0.8182 | 0.7478 |
| pima_diabetes | 0.2000 | A1_FaNet_NoLoss | 0.6308 | 0.0386 | 0.7068 | 0.0371 | 0.8178 | 0.7528 |
| pima_diabetes | 0.2000 | A3_FaNet_Permuted | 0.6141 | 0.0636 | 0.6946 | 0.0425 | 0.8166 | 0.7536 |
| pima_diabetes | 0.2000 | B1_RF_Reg | 0.6274 | 0.0465 | 0.6913 | 0.0427 | 0.8216 | 0.7627 |
| pima_diabetes | 0.2000 | B2_HistGB | 0.6099 | 0.0334 | 0.6552 | 0.0525 | 0.8018 | 0.7442 |
| pima_diabetes | 0.2000 | F1_FaNet_Full | 0.6115 | 0.0517 | 0.6988 | 0.0386 | 0.8178 | 0.7518 |
| pima_diabetes | 0.2000 | M0_MLP_Iso | 0.6111 | 0.0675 | 0.6835 | 0.0534 | 0.8136 | 0.7457 |
| pima_diabetes | 0.3000 | A1_FaNet_NoLoss | 0.6314 | 0.0265 | 0.7048 | 0.0318 | 0.8159 | 0.7505 |
| pima_diabetes | 0.3000 | A3_FaNet_Permuted | 0.6119 | 0.0563 | 0.6924 | 0.0352 | 0.8136 | 0.7481 |
| pima_diabetes | 0.3000 | B1_RF_Reg | 0.6287 | 0.0342 | 0.6866 | 0.0339 | 0.8213 | 0.7627 |
| pima_diabetes | 0.3000 | B2_HistGB | 0.6140 | 0.0344 | 0.6549 | 0.0393 | 0.8011 | 0.7448 |
| pima_diabetes | 0.3000 | F1_FaNet_Full | 0.6131 | 0.0484 | 0.6987 | 0.0309 | 0.8160 | 0.7497 |
| pima_diabetes | 0.3000 | M0_MLP_Iso | 0.6122 | 0.0630 | 0.6816 | 0.0450 | 0.8098 | 0.7447 |
| pima_diabetes | 0.5000 | A1_FaNet_NoLoss | 0.6253 | 0.0520 | 0.6701 | 0.0508 | 0.7953 | 0.7448 |
| pima_diabetes | 0.5000 | A3_FaNet_Permuted | 0.5948 | 0.0633 | 0.6627 | 0.0480 | 0.7939 | 0.7338 |
| pima_diabetes | 0.5000 | B1_RF_Reg | 0.6070 | 0.0520 | 0.6560 | 0.0547 | 0.7971 | 0.7445 |
| pima_diabetes | 0.5000 | B2_HistGB | 0.5923 | 0.0426 | 0.6302 | 0.0565 | 0.7770 | 0.7281 |
| pima_diabetes | 0.5000 | F1_FaNet_Full | 0.5929 | 0.0611 | 0.6645 | 0.0515 | 0.7949 | 0.7362 |
| pima_diabetes | 0.5000 | M0_MLP_Iso | 0.5985 | 0.0727 | 0.6549 | 0.0527 | 0.7918 | 0.7317 |

## Exp 3 : Généralisation OOD sans fuite

| domain | variant | recall_mean | recall_std | f1_mean | f1_std | roc_auc_mean | pr_auc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IN_rice | A1_FaNet_NoLoss | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| IN_rice | A3_FaNet_Permuted | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| IN_rice | B0_LogReg | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| IN_rice | B1_RF_Reg | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| IN_rice | B2_HistGB | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| IN_rice | F1_FaNet_Full | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| IN_rice | M0_MLP_Iso | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| OOD_jute | A1_FaNet_NoLoss | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| OOD_jute | A3_FaNet_Permuted | 0.9680 | 0.0438 | 0.9833 | 0.0228 | 1.0000 | 1.0000 |
| OOD_jute | B0_LogReg | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| OOD_jute | B1_RF_Reg | 0.9180 | 0.0084 | 0.9572 | 0.0045 | 1.0000 | 1.0000 |
| OOD_jute | B2_HistGB | 0.7100 | 0.0000 | 0.8304 | 0.0000 | 0.8550 | 0.8418 |
| OOD_jute | F1_FaNet_Full | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| OOD_jute | M0_MLP_Iso | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| OOD_maize | A1_FaNet_NoLoss | 0.4960 | 0.1184 | 0.6564 | 0.1067 | 1.0000 | 1.0000 |
| OOD_maize | A3_FaNet_Permuted | 0.1360 | 0.1457 | 0.2169 | 0.2207 | 0.9999 | 0.9999 |
| OOD_maize | B0_LogReg | 0.2840 | 0.0344 | 0.4415 | 0.0426 | 1.0000 | 1.0000 |
| OOD_maize | B1_RF_Reg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9981 | 0.9945 |
| OOD_maize | B2_HistGB | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.4545 |
| OOD_maize | F1_FaNet_Full | 0.4700 | 0.2581 | 0.6048 | 0.2484 | 1.0000 | 0.9999 |
| OOD_maize | M0_MLP_Iso | 0.7680 | 0.1558 | 0.8617 | 0.1003 | 1.0000 | 1.0000 |

## Exp 4 : Calibration clinique (ECE pondérée standard)

| dataset | variant | brier_score | weighted_ece | min_prob | max_prob |
| --- | --- | --- | --- | --- | --- |
| pima_diabetes | B0_LogReg | 0.1580 | 0.0329 | 0.0017 | 0.9940 |
| pima_diabetes | B1_RF_Reg | 0.1595 | 0.0315 | 0.0014 | 0.9039 |
| pima_diabetes | B2_HistGB | 0.1683 | 0.0592 | 0.0025 | 0.9815 |
| pima_diabetes | M0_MLP_Iso | 0.1652 | 0.0647 | 0.0000 | 0.9888 |
| pima_diabetes | A4_FaNet_FaDual | 0.1593 | 0.0173 | 0.0232 | 0.8904 |
| pima_diabetes | F1_FaNet_Full | 0.1601 | 0.0396 | 0.0098 | 0.8795 |
| breast_cancer | B0_LogReg | 0.0199 | 0.0150 | 0.0000 | 1.0000 |
| breast_cancer | B1_RF_Reg | 0.0346 | 0.0359 | 0.0000 | 1.0000 |
| breast_cancer | B2_HistGB | 0.0272 | 0.0214 | 0.0002 | 0.9998 |
| breast_cancer | M0_MLP_Iso | 0.0270 | 0.0291 | 0.0000 | 1.0000 |
| breast_cancer | A4_FaNet_FaDual | 0.0296 | 0.0236 | 0.0177 | 0.9820 |
| breast_cancer | F1_FaNet_Full | 0.0268 | 0.0127 | 0.0011 | 0.9995 |

## Exp 5 : Régression (Diabetes réel & Prix synthétiques)

| dataset | variant | mae_mean | mae_std | rmse_mean | r2_mean | r2_std |
| --- | --- | --- | --- | --- | --- | --- |
| agri_price_synthetic | A1_FaNet_NoLoss | 43.1103 | 1.7856 | 54.1879 | 0.9471 | 0.0055 |
| agri_price_synthetic | B0_Ridge | 104.8682 | 3.7602 | 131.8076 | 0.6882 | 0.0179 |
| agri_price_synthetic | B1_RF_Reg | 43.4118 | 1.1705 | 54.4502 | 0.9465 | 0.0056 |
| agri_price_synthetic | F1_FaNet_Full | 42.8961 | 1.6444 | 53.9482 | 0.9476 | 0.0051 |
| agri_price_synthetic | M0_MLP_Iso | 42.7946 | 1.4033 | 53.7194 | 0.9480 | 0.0057 |
| sklearn_diabetes_real | A1_FaNet_NoLoss | 45.3021 | 3.2197 | 55.9581 | 0.4591 | 0.0731 |
| sklearn_diabetes_real | B0_Ridge | 44.3238 | 2.4951 | 54.8143 | 0.4802 | 0.0726 |
| sklearn_diabetes_real | B1_RF_Reg | 46.9443 | 2.7200 | 57.4816 | 0.4298 | 0.0748 |
| sklearn_diabetes_real | F1_FaNet_Full | 45.3201 | 3.2962 | 56.1182 | 0.4561 | 0.0740 |
| sklearn_diabetes_real | M0_MLP_Iso | 45.0986 | 2.6527 | 55.8129 | 0.4610 | 0.0798 |

## Exp 6 : Séries temporelles IoT (Purge 18h + BiLSTM baseline)

| variant | f1_mean | f1_std | recall_mean | precision_mean | roc_auc_mean | pr_auc_mean |
| --- | --- | --- | --- | --- | --- | --- |
| A1_FaNet_BiLSTM_NoLoss | 0.8019 | 0.0629 | 0.8263 | 0.8024 | 0.9854 | 0.8063 |
| B1_RF_Reg | 0.9247 | 0.0124 | 1.0000 | 0.8602 | 0.9945 | 0.8971 |
| B2_HistGB | 0.9114 | 0.0000 | 0.9474 | 0.8780 | 0.9911 | 0.8309 |
| F1_FaNet_BiLSTM_Full | 0.8488 | 0.0605 | 0.9316 | 0.7824 | 0.9870 | 0.8070 |
| M0_BiLSTM_Iso | 0.6956 | 0.0621 | 0.6211 | 0.8235 | 0.9549 | 0.6921 |

## Exp 7 : Audit MIA Confidentialité vs Utilité

| variant | test_roc_auc_mean | test_f1_mean | mia_loss_auc_mean | mia_loss_auc_std | mia_conf_auc_mean | tpr_1pct_mean |
| --- | --- | --- | --- | --- | --- | --- |
| A1_FaNet_NoLoss | 0.8248 | 0.6386 | 0.5133 | 0.0134 | 0.5052 | 0.0153 |
| A4_FaNet_FaDual | 0.8175 | 0.6297 | 0.5265 | 0.0110 | 0.5078 | 0.0138 |
| B0_LogReg | 0.8238 | 0.6293 | 0.5133 | 0.0195 | 0.5130 | 0.0174 |
| B1_RF_Reg | 0.8172 | 0.6173 | 0.5895 | 0.0168 | 0.5452 | 0.0179 |
| F1_FaNet_Full | 0.8187 | 0.6350 | 0.5229 | 0.0146 | 0.5078 | 0.0128 |
| M0_MLP_Iso | 0.8152 | 0.6307 | 0.5337 | 0.0233 | 0.5058 | 0.0153 |
| RF_Unreg | 0.8099 | 0.6258 | 0.7674 | 0.0193 | 0.7409 | 0.0308 |

## Exp 8 : Efficacité, Latence par patient & ONNX

| model | params | train_s | infer_ms_per_1k | nWh_per_patient | train_uWh | f1 |
| --- | --- | --- | --- | --- | --- | --- |
| B1_RF_Reg | -1 | 0.3789 | 47.3684 | 880.4786 | 7399.0051 | 0.5771 |
| B2_HistGB | -1 | 0.1137 | 12.3167 | 242.5595 | 2061.6935 | 0.5743 |
| M0_MLP_Iso | 2433 | 0.8764 | 0.6064 | 12.6302 | 16793.5359 | 0.5672 |
| F1_FaNet_PyTorch | 2361 | 3.4750 | 6.5247 | 124.3555 | 65622.1687 | 0.6449 |
| F1_FaNet_ONNX | 2361 | 3.4750 | 12.3579 | 230.2581 | 65622.1687 | 0.6449 |
