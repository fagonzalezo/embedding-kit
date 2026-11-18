
### pets  train [3680, 2048] test [3669, 2048]  info={'note': 'ResNet50 ImageNet features, Oxford-IIIT Pets'}

| method | dim | kNN-euc | kNN-cos | LinProbe | Trust@10 | Cont@10 | k-skew@10 | EffRank | fit s |
|---|---|---|---|---|---|---|---|---|---|
| raw | 2048 | 0.898 | 0.901 | 0.916 | 1.000 | 1.000 | 1.86 | 154.5 | 0.0 |
| centre+L2 | 2048 | 0.900 | 0.900 | 0.912 | 0.998 | 0.999 | 1.58 | 147.7 | 0.0 |
| standardise | 2048 | 0.829 | 0.839 | 0.916 | 0.969 | 0.941 | 8.47 | 603.3 | 0.0 |
| PCA-16 | 16 | 0.891 | 0.891 | 0.900 | 0.988 | 0.995 | 0.80 | 13.6 | 0.1 |
| PCA-32 | 32 | 0.899 | 0.903 | 0.910 | 0.994 | 0.997 | 0.89 | 24.0 | 0.1 |
| PCA-64 | 64 | 0.907 | 0.902 | 0.914 | 0.997 | 0.999 | 1.14 | 37.0 | 0.1 |
| PCA-128 | 128 | 0.903 | 0.900 | 0.911 | 0.999 | 0.999 | 1.31 | 54.3 | 0.2 |
| PCA-whiten-16 | 16 | 0.886 | 0.887 | 0.900 | 0.987 | 0.994 | 0.78 | 15.9 | 0.1 |
| PCA-whiten-32 | 32 | 0.893 | 0.887 | 0.910 | 0.993 | 0.997 | 0.87 | 31.1 | 0.1 |
| PCA-whiten-64 | 64 | 0.889 | 0.885 | 0.914 | 0.996 | 0.998 | 1.24 | 60.5 | 0.1 |
| PCA-whiten-128 | 128 | 0.867 | 0.866 | 0.911 | 0.992 | 0.996 | 1.72 | 117.6 | 0.2 |
| ABTT-D1 | 2048 | 0.897 | 0.895 | 0.915 | 1.000 | 1.000 | 1.91 | 181.2 | 0.1 |
| ABTT-D3 | 2048 | 0.894 | 0.889 | 0.914 | 1.000 | 1.000 | 1.95 | 205.3 | 0.1 |
| ABTT-D20 | 2048 | 0.679 | 0.837 | 0.873 | 0.897 | 0.981 | 18.81 | 369.1 | 0.1 |
| EmbedKit-default (n=3) | 22 | 0.893 ± 0.003 | 0.893 ± 0.003 | 0.906 ± 0.004 | 0.987 ± 0.000 | 0.994 ± 0.000 | 0.55 ± 0.04 | 20.5 ± 0.3 | 23.9 ± 1.2 |
| EmbedKit-td64-noES-80ep (n=3) | 64 | 0.898 ± 0.004 | 0.898 ± 0.004 | 0.905 ± 0.002 | 0.988 ± 0.001 | 0.995 ± 0.000 | 0.53 ± 0.08 | 34.0 ± 0.5 | 34.9 ± 5.2 |
| EmbedKit-supervised (n=3) | 22 | 0.918 ± 0.002 | 0.918 ± 0.002 | 0.920 ± 0.003 | 0.986 ± 0.000 | 0.992 ± 0.000 | 1.17 ± 0.01 | 16.4 ± 0.0 | 21.3 ± 3.7 |

EmbedKit run details:
- EmbedKit-default seed 0: target_dim=22, stopped_epoch=55/100, ID estimates={'TwoNN': 16.15, 'MLE': 12.86, 'lPCA': 38.0, 'MOM': 9.38}, monitored k_skewness history (every 5 ep)=[0.071, 0.415, 0.324, 0.442, 0.393, 0.315, 0.412, 0.193, 0.169, 0.25, 0.183], final loss=0.042, warnings=['RuntimeWarning: faiss import failed — falling back to sklearn kNN, which is much slower. Check your faiss-cpu installation.']
- EmbedKit-default seed 1: target_dim=22, stopped_epoch=55/100, ID estimates={'TwoNN': 16.15, 'MLE': 12.86, 'lPCA': 38.0, 'MOM': 9.38}, monitored k_skewness history (every 5 ep)=[-0.079, 0.008, -0.034, -0.005, 0.245, 0.188, 0.395, 0.256, 0.264, 0.325, 0.22], final loss=0.029, warnings=[]
- EmbedKit-default seed 2: target_dim=22, stopped_epoch=55/100, ID estimates={'TwoNN': 16.15, 'MLE': 12.86, 'lPCA': 38.0, 'MOM': 9.38}, monitored k_skewness history (every 5 ep)=[0.159, 0.306, 0.31, 0.428, 0.459, 0.423, 0.42, 0.314, 0.257, 0.312, 0.29], final loss=0.031, warnings=[]
- EmbedKit-td64-noES-80ep seed 0: target_dim=64, stopped_epoch=80/80, ID estimates={'TwoNN': 16.15, 'MLE': 12.86, 'lPCA': 38.0, 'MOM': 9.38}, monitored k_skewness history (every 5 ep)=[0.609, 0.397, 0.414, 0.34, 0.349, 0.38, 0.347, 0.367, 0.349, 0.343, 0.343, 0.344, 0.325, 0.323, 0.318, 0.315], final loss=-0.082, warnings=[]
- EmbedKit-td64-noES-80ep seed 1: target_dim=64, stopped_epoch=80/80, ID estimates={'TwoNN': 16.15, 'MLE': 12.86, 'lPCA': 38.0, 'MOM': 9.38}, monitored k_skewness history (every 5 ep)=[0.534, 0.402, 0.363, 0.356, 0.343, 0.28, 0.29, 0.264, 0.301, 0.288, 0.254, 0.251, 0.259, 0.25, 0.253, 0.274], final loss=-0.099, warnings=[]
- EmbedKit-td64-noES-80ep seed 2: target_dim=64, stopped_epoch=80/80, ID estimates={'TwoNN': 16.15, 'MLE': 12.86, 'lPCA': 38.0, 'MOM': 9.38}, monitored k_skewness history (every 5 ep)=[0.515, 0.385, 0.382, 0.351, 0.298, 0.253, 0.252, 0.254, 0.276, 0.341, 0.263, 0.284, 0.299, 0.294, 0.285, 0.312], final loss=-0.096, warnings=[]
- EmbedKit-supervised seed 0: target_dim=22, stopped_epoch=65/100, ID estimates={'TwoNN': 16.15, 'MLE': 12.86, 'lPCA': 38.0, 'MOM': 9.38}, monitored k_skewness history (every 5 ep)=[0.509, 0.449, 0.287, 0.447, 0.554, 0.476, 0.442, 0.384, 0.485, 0.523, 0.399, 0.429, 0.483], final loss=1.576, warnings=[]
- EmbedKit-supervised seed 1: target_dim=22, stopped_epoch=65/100, ID estimates={'TwoNN': 16.15, 'MLE': 12.86, 'lPCA': 38.0, 'MOM': 9.38}, monitored k_skewness history (every 5 ep)=[0.287, 0.219, 0.145, 0.26, 0.33, 0.276, 0.234, 0.334, 0.314, 0.354, 0.488, 0.398, 0.356], final loss=1.572, warnings=[]
- EmbedKit-supervised seed 2: target_dim=22, stopped_epoch=55/100, ID estimates={'TwoNN': 16.15, 'MLE': 12.86, 'lPCA': 38.0, 'MOM': 9.38}, monitored k_skewness history (every 5 ep)=[-0.048, -0.008, 0.051, 0.086, 0.201, 0.065, 0.04, 0.029, 0.002, 0.035, -0.016], final loss=1.568, warnings=[]

Baseline warnings: []

### digits  train [898, 64] test [899, 64]  info={'note': 'raw 8x8 pixels, NOT a learned embedding'}

| method | dim | kNN-euc | kNN-cos | LinProbe | Trust@10 | Cont@10 | k-skew@10 | EffRank | fit s |
|---|---|---|---|---|---|---|---|---|---|
| raw | 64 | 0.984 | 0.981 | 0.963 | 1.000 | 1.000 | 0.72 | 20.4 | 0.0 |
| centre+L2 | 64 | 0.972 | 0.972 | 0.960 | 0.999 | 1.000 | 0.67 | 20.7 | 0.0 |
| standardise | 64 | 0.963 | 0.949 | 0.963 | 0.996 | 0.982 | 0.87 | 16.1 | 0.0 |
| PCA-16 | 16 | 0.981 | 0.972 | 0.944 | 0.999 | 0.999 | 0.69 | 11.9 | 0.0 |
| PCA-32 | 32 | 0.981 | 0.976 | 0.948 | 1.000 | 1.000 | 0.68 | 17.6 | 0.0 |
| PCA-64 | 64 | 0.984 | 0.972 | 0.934 | 1.000 | 1.000 | 0.72 | 20.4 | 0.0 |
| PCA-whiten-16 | 16 | 0.971 | 0.970 | 0.944 | 0.996 | 0.996 | 0.71 | 15.7 | 0.0 |
| PCA-whiten-32 | 32 | 0.966 | 0.966 | 0.948 | 0.988 | 0.991 | 1.15 | 30.7 | 0.0 |
| PCA-whiten-64 (capped to rank 59) | 59 | 0.905 | 0.941 | 0.935 | 0.958 | 0.932 | 2.08 | 23.2 | 0.0 |
| ABTT-D1 | 64 | 0.974 | 0.960 | 0.928 | 0.992 | 1.000 | 0.82 | 21.1 | 0.0 |
| ABTT-D3 | 64 | 0.953 | 0.963 | 0.861 | 0.959 | 0.997 | 1.30 | 24.2 | 0.0 |
| EmbedKit-default (n=3) | 11 | 0.966 ± 0.006 | 0.966 ± 0.006 | 0.931 ± 0.004 | 0.993 ± 0.001 | 0.995 ± 0.000 | 0.31 ± 0.04 | 10.0 ± 0.1 | 2.2 ± 0.2 |
| EmbedKit-td64-noES-80ep (n=3) | 64 | 0.976 ± 0.003 | 0.976 ± 0.003 | 0.966 ± 0.002 | 0.996 ± 0.000 | 0.997 ± 0.000 | 0.30 ± 0.03 | 21.3 ± 0.2 | 2.0 ± 0.0 |
| EmbedKit-supervised (n=3) | 9/11 | 0.977 ± 0.004 | 0.977 ± 0.004 | 0.976 ± 0.003 | 0.935 ± 0.006 | 0.973 ± 0.001 | 0.81 ± 0.08 | 6.3 ± 0.2 | 1.8 ± 0.3 |

EmbedKit run details:
- EmbedKit-default seed 0: target_dim=11, stopped_epoch=55/100, ID estimates={'TwoNN': 8.23, 'MLE': 6.23, 'lPCA': 23.0, 'MOM': 6.17}, monitored k_skewness history (every 5 ep)=[0.168, 0.415, 0.362, 0.438, 0.614, 0.493, 0.477, 0.441, 0.449, 0.477, 0.439], final loss=1.213, warnings=['RuntimeWarning: divide by zero encountered in divide', 'RuntimeWarning: faiss import failed — falling back to sklearn kNN, which is much slower. Check your faiss-cpu installation.']
- EmbedKit-default seed 1: target_dim=11, stopped_epoch=85/100, ID estimates={'TwoNN': 8.21, 'MLE': 6.23, 'lPCA': 23.0, 'MOM': 6.17}, monitored k_skewness history (every 5 ep)=[0.709, 0.39, 0.317, 0.266, 0.254, 0.28, 0.133, 0.246, 0.262, 0.237, 0.224, 0.27, 0.201, 0.299, 0.338, 0.343, 0.4], final loss=1.217, warnings=['RuntimeWarning: divide by zero encountered in divide']
- EmbedKit-default seed 2: target_dim=11, stopped_epoch=100/100, ID estimates={'TwoNN': 8.21, 'MLE': 6.23, 'lPCA': 23.0, 'MOM': 6.17}, monitored k_skewness history (every 5 ep)=[0.188, 0.567, 0.625, 0.578, 0.318, 0.447, 0.276, 0.175, 0.27, 0.309, 0.291, 0.399, 0.18, 0.243, 0.274, 0.15, 0.315, 0.15, 0.217, 0.183], final loss=1.252, warnings=['RuntimeWarning: divide by zero encountered in divide']
- EmbedKit-td64-noES-80ep seed 0: target_dim=64, stopped_epoch=80/80, ID estimates={'TwoNN': 8.21, 'MLE': 6.23, 'lPCA': 23.0, 'MOM': 6.17}, monitored k_skewness history (every 5 ep)=[0.405, 0.357, 0.277, 0.244, 0.289, 0.249, 0.217, 0.2, 0.18, 0.182, 0.215, 0.183, 0.184, 0.178, 0.199, 0.203], final loss=0.954, warnings=['RuntimeWarning: divide by zero encountered in divide']
- EmbedKit-td64-noES-80ep seed 1: target_dim=64, stopped_epoch=80/80, ID estimates={'TwoNN': 8.21, 'MLE': 6.23, 'lPCA': 23.0, 'MOM': 6.17}, monitored k_skewness history (every 5 ep)=[0.508, 0.39, 0.329, 0.275, 0.294, 0.302, 0.288, 0.304, 0.315, 0.282, 0.308, 0.296, 0.305, 0.302, 0.312, 0.305], final loss=0.996, warnings=['RuntimeWarning: divide by zero encountered in divide']
- EmbedKit-td64-noES-80ep seed 2: target_dim=64, stopped_epoch=80/80, ID estimates={'TwoNN': nan, 'MLE': 6.23, 'lPCA': 23.0, 'MOM': 6.17}, monitored k_skewness history (every 5 ep)=[0.495, 0.343, 0.307, 0.294, 0.287, 0.267, 0.261, 0.294, 0.26, 0.287, 0.292, 0.301, 0.27, 0.309, 0.312, 0.313], final loss=0.988, warnings=['RuntimeWarning: divide by zero encountered in divide', 'RuntimeWarning: invalid value encountered in divide', 'UserWarning: ID method TwoNN failed: Input X contains NaN.\nLinearRegression does not accept missing values encoded as NaN natively. For supervised learning, you might want t']
- EmbedKit-supervised seed 0: target_dim=9, stopped_epoch=60/100, ID estimates={'TwoNN': 5.95, 'MLE': 6.23, 'lPCA': 23.0, 'MOM': 6.17}, monitored k_skewness history (every 5 ep)=[0.322, 0.184, 0.197, 0.221, 0.226, 0.305, 0.283, 0.336, 0.251, 0.201, 0.244, 0.325], final loss=3.303, warnings=['RuntimeWarning: divide by zero encountered in divide', 'RuntimeWarning: invalid value encountered in divide']
- EmbedKit-supervised seed 1: target_dim=11, stopped_epoch=85/100, ID estimates={'TwoNN': 8.22, 'MLE': 6.23, 'lPCA': 23.0, 'MOM': 6.17}, monitored k_skewness history (every 5 ep)=[0.673, 0.609, 0.675, 0.661, 0.628, 0.665, 0.523, 0.592, 0.561, 0.534, 0.618, 0.566, 0.599, 0.618, 0.586, 0.55, 0.59], final loss=3.264, warnings=['RuntimeWarning: divide by zero encountered in divide']
- EmbedKit-supervised seed 2: target_dim=9, stopped_epoch=65/100, ID estimates={'TwoNN': nan, 'MLE': 0.0, 'lPCA': 23.0, 'MOM': 6.17}, monitored k_skewness history (every 5 ep)=[0.449, 0.399, 0.152, 0.455, 0.281, 0.449, 0.471, 0.586, 0.591, 0.584, 0.653, 0.7, 0.779], final loss=3.339, warnings=['RuntimeWarning: divide by zero encountered in divide', 'RuntimeWarning: invalid value encountered in divide', 'UserWarning: ID method TwoNN failed: Input X contains NaN.\nLinearRegression does not accept missing values encoded as NaN natively. For supervised learning, you might want t']

Baseline warnings: []

### synthetic  train [3000, 512] test [3000, 512]  info={'sig_std': 0.5039322842317635, 'dev_norm': 11.459604440057792, 'mean_norm': 229.19208880115585, 'sep': 1.0, 'mean_scale': 20.0, 'nuis_scale': 2.0}

| method | dim | kNN-euc | kNN-cos | LinProbe | Trust@10 | Cont@10 | k-skew@10 | EffRank | fit s |
|---|---|---|---|---|---|---|---|---|---|
| raw | 512 | 0.744 | 0.746 | 0.933 | 1.000 | 1.000 | 0.61 | 7.5 | 0.0 |
| centre+L2 | 512 | 0.741 | 0.741 | 0.930 | 0.941 | 0.994 | 1.02 | 10.0 | 0.0 |
| standardise | 512 | 0.789 | 0.781 | 0.933 | 0.999 | 0.999 | 0.79 | 10.2 | 0.0 |
| PCA-16 | 16 | 0.730 | 0.745 | 0.927 | 1.000 | 1.000 | 0.52 | 5.9 | 0.0 |
| PCA-32 | 32 | 0.747 | 0.748 | 0.936 | 1.000 | 1.000 | 0.39 | 6.5 | 0.0 |
| PCA-64 | 64 | 0.748 | 0.742 | 0.930 | 1.000 | 1.000 | 0.47 | 6.9 | 0.0 |
| PCA-128 | 128 | 0.745 | 0.741 | 0.910 | 1.000 | 1.000 | 0.56 | 7.2 | 0.1 |
| PCA-whiten-16 | 16 | 0.874 | 0.874 | 0.927 | 0.904 | 0.946 | 1.36 | 15.9 | 0.0 |
| PCA-whiten-32 | 32 | 0.879 | 0.890 | 0.936 | 0.794 | 0.872 | 1.76 | 31.7 | 0.0 |
| PCA-whiten-64 | 64 | 0.868 | 0.893 | 0.930 | 0.709 | 0.779 | 5.14 | 62.6 | 0.0 |
| PCA-whiten-128 | 128 | 0.852 | 0.908 | 0.910 | 0.645 | 0.702 | 13.77 | 122.4 | 0.1 |
| ABTT-D1 | 512 | 0.828 | 0.825 | 0.932 | 0.881 | 0.994 | 0.79 | 7.0 | 0.0 |
| ABTT-D3 | 512 | 0.919 | 0.913 | 0.931 | 0.602 | 0.896 | 1.53 | 24.2 | 0.0 |
| ABTT-D5 | 512 | 0.846 | 0.833 | 0.892 | 0.590 | 0.826 | 1.59 | 27.1 | 0.0 |
| EmbedKit-default (n=3) | 15 | 0.733 ± 0.008 | 0.733 ± 0.008 | 0.869 ± 0.007 | 0.996 ± 0.000 | 0.998 ± 0.000 | 0.25 ± 0.04 | 14.0 ± 0.2 | 14.7 ± 2.7 |
| EmbedKit-td64-noES-80ep (n=3) | 64 | 0.764 ± 0.004 | 0.764 ± 0.004 | 0.900 ± 0.006 | 0.997 ± 0.000 | 0.998 ± 0.000 | 0.27 ± 0.04 | 20.6 ± 0.2 | 19.5 ± 2.5 |
| EmbedKit-supervised (n=3) | 15 | 0.928 ± 0.002 | 0.928 ± 0.002 | 0.933 ± 0.003 | 0.912 ± 0.002 | 0.903 ± 0.001 | 0.98 ± 0.02 | 9.3 ± 0.1 | 15.1 ± 1.3 |

EmbedKit run details:
- EmbedKit-default seed 0: target_dim=15, stopped_epoch=90/100, ID estimates={'TwoNN': 12.62, 'MLE': 10.52, 'lPCA': 8.0, 'MOM': 9.31}, monitored k_skewness history (every 5 ep)=[0.147, 0.135, 0.358, 0.343, 0.17, 0.18, 0.151, -0.044, 0.094, 0.039, 0.091, 0.063, 0.118, 0.002, -0.033, -0.037, 0.007, 0.075], final loss=0.246, warnings=['RuntimeWarning: faiss import failed — falling back to sklearn kNN, which is much slower. Check your faiss-cpu installation.']
- EmbedKit-default seed 1: target_dim=15, stopped_epoch=100/100, ID estimates={'TwoNN': 12.62, 'MLE': 10.52, 'lPCA': 8.0, 'MOM': 9.31}, monitored k_skewness history (every 5 ep)=[0.423, 0.457, 0.553, 0.333, 0.412, 0.329, 0.231, 0.294, 0.246, 0.259, 0.336, 0.229, 0.189, 0.232, 0.137, 0.202, 0.252, 0.213, 0.248, 0.241], final loss=0.277, warnings=[]
- EmbedKit-default seed 2: target_dim=15, stopped_epoch=55/100, ID estimates={'TwoNN': 12.62, 'MLE': 10.52, 'lPCA': 8.0, 'MOM': 9.31}, monitored k_skewness history (every 5 ep)=[-0.122, 0.154, -0.008, 0.059, 0.142, 0.063, 0.115, 0.028, 0.188, 0.377, 0.232], final loss=0.276, warnings=[]
- EmbedKit-td64-noES-80ep seed 0: target_dim=64, stopped_epoch=80/80, ID estimates={'TwoNN': 12.62, 'MLE': 10.52, 'lPCA': 8.0, 'MOM': 9.31}, monitored k_skewness history (every 5 ep)=[0.259, 0.279, 0.321, 0.305, 0.265, 0.312, 0.258, 0.265, 0.23, 0.263, 0.225, 0.23, 0.23, 0.241, 0.241, 0.224], final loss=0.190, warnings=[]
- EmbedKit-td64-noES-80ep seed 1: target_dim=64, stopped_epoch=80/80, ID estimates={'TwoNN': 12.62, 'MLE': 10.52, 'lPCA': 8.0, 'MOM': 9.31}, monitored k_skewness history (every 5 ep)=[0.232, 0.246, 0.242, 0.273, 0.275, 0.297, 0.265, 0.235, 0.246, 0.286, 0.241, 0.27, 0.262, 0.268, 0.265, 0.272], final loss=0.194, warnings=[]
- EmbedKit-td64-noES-80ep seed 2: target_dim=64, stopped_epoch=80/80, ID estimates={'TwoNN': 12.62, 'MLE': 10.52, 'lPCA': 8.0, 'MOM': 9.31}, monitored k_skewness history (every 5 ep)=[0.287, 0.297, 0.217, 0.205, 0.202, 0.238, 0.212, 0.232, 0.231, 0.276, 0.242, 0.243, 0.205, 0.228, 0.225, 0.225], final loss=0.184, warnings=[]
- EmbedKit-supervised seed 0: target_dim=15, stopped_epoch=65/100, ID estimates={'TwoNN': 12.62, 'MLE': 10.52, 'lPCA': 8.0, 'MOM': 9.31}, monitored k_skewness history (every 5 ep)=[0.586, 0.335, 0.233, 0.314, 0.425, 0.443, 0.295, 0.315, 0.331, 0.343, 0.344, 0.394, 0.374], final loss=4.812, warnings=[]
- EmbedKit-supervised seed 1: target_dim=15, stopped_epoch=60/100, ID estimates={'TwoNN': 12.62, 'MLE': 10.52, 'lPCA': 8.0, 'MOM': 9.31}, monitored k_skewness history (every 5 ep)=[0.4, 0.291, 0.371, 0.301, 0.387, 0.389, 0.524, 0.385, 0.372, 0.352, 0.297, 0.417], final loss=4.782, warnings=[]
- EmbedKit-supervised seed 2: target_dim=15, stopped_epoch=100/100, ID estimates={'TwoNN': 12.62, 'MLE': 10.52, 'lPCA': 8.0, 'MOM': 9.31}, monitored k_skewness history (every 5 ep)=[0.63, 0.628, 0.538, 0.509, 0.532, 0.568, 0.538, 0.504, 0.49, 0.451, 0.492, 0.482, 0.501, 0.545, 0.503, 0.475, 0.474, 0.486, 0.482, 0.482], final loss=4.754, warnings=[]

Baseline warnings: []

### Step 4 trajectories (synthetic)


seed 0 target_dim=15 rho(kskew,acc)=[0.40722645453331024, 0.07473675781585976] rho(unif,acc)=[-0.8611221145769075, 1.0904054271806364e-06] argmin_kskew_ep=10 argmax_acc_ep=80 sim_ES_stop=60 acc@stop=0.7147

| epoch | loss | k-skew (held-out) | uniformity (held-out) | test kNN acc |
|---|---|---|---|---|
| 5 | 0.959 | 0.296 | -3.001 | 0.6707 |
| 10 | 0.653 | 0.017 | -3.140 | 0.6930 |
| 15 | 0.565 | 0.126 | -3.221 | 0.6987 |
| 20 | 0.509 | 0.226 | -3.244 | 0.7017 |
| 25 | 0.416 | 0.204 | -3.262 | 0.7003 |
| 30 | 0.388 | 0.236 | -3.278 | 0.7090 |
| 35 | 0.333 | 0.068 | -3.275 | 0.7063 |
| 40 | 0.339 | 0.216 | -3.286 | 0.7130 |
| 45 | 0.355 | 0.273 | -3.282 | 0.7147 |
| 50 | 0.315 | 0.338 | -3.287 | 0.7213 |
| 55 | 0.259 | 0.195 | -3.284 | 0.7210 |
| 60 | 0.262 | 0.145 | -3.284 | 0.7147 |
| 65 | 0.283 | 0.202 | -3.293 | 0.7107 |
| 70 | 0.286 | 0.298 | -3.294 | 0.7203 |
| 75 | 0.272 | 0.307 | -3.293 | 0.7207 |
| 80 | 0.243 | 0.282 | -3.292 | 0.7227 |
| 85 | 0.256 | 0.259 | -3.296 | 0.7210 |
| 90 | 0.260 | 0.207 | -3.296 | 0.7217 |
| 95 | 0.258 | 0.247 | -3.295 | 0.7227 |
| 100 | 0.259 | 0.225 | -3.296 | 0.7220 |

seed 1 target_dim=15 rho(kskew,acc)=[-0.37457693510179774, 0.10370016887863298] rho(unif,acc)=[-0.8424220227189027, 3.1637218417635983e-06] argmin_kskew_ep=50 argmax_acc_ep=95 sim_ES_stop=100 acc@stop=0.7147

| epoch | loss | k-skew (held-out) | uniformity (held-out) | test kNN acc |
|---|---|---|---|---|
| 5 | 0.974 | 0.464 | -2.980 | 0.6977 |
| 10 | 0.718 | 0.366 | -3.108 | 0.7053 |
| 15 | 0.582 | 0.422 | -3.209 | 0.7023 |
| 20 | 0.498 | 0.263 | -3.240 | 0.6930 |
| 25 | 0.453 | 0.292 | -3.257 | 0.6923 |
| 30 | 0.391 | 0.234 | -3.267 | 0.7007 |
| 35 | 0.357 | 0.284 | -3.273 | 0.6940 |
| 40 | 0.361 | 0.281 | -3.279 | 0.7090 |
| 45 | 0.337 | 0.284 | -3.287 | 0.7087 |
| 50 | 0.350 | 0.196 | -3.281 | 0.7043 |
| 55 | 0.314 | 0.271 | -3.287 | 0.7137 |
| 60 | 0.329 | 0.268 | -3.289 | 0.7153 |
| 65 | 0.296 | 0.317 | -3.292 | 0.7097 |
| 70 | 0.287 | 0.261 | -3.294 | 0.7150 |
| 75 | 0.304 | 0.297 | -3.292 | 0.7133 |
| 80 | 0.257 | 0.284 | -3.291 | 0.7130 |
| 85 | 0.304 | 0.301 | -3.293 | 0.7120 |
| 90 | 0.235 | 0.234 | -3.292 | 0.7137 |
| 95 | 0.265 | 0.244 | -3.294 | 0.7157 |
| 100 | 0.280 | 0.220 | -3.294 | 0.7147 |

seed 2 target_dim=14 rho(kskew,acc)=[0.7720092478912439, 6.673664566063697e-05] rho(unif,acc)=[-0.9541010977837204, 7.266212437796004e-11] argmin_kskew_ep=5 argmax_acc_ep=95 sim_ES_stop=55 acc@stop=0.7203

| epoch | loss | k-skew (held-out) | uniformity (held-out) | test kNN acc |
|---|---|---|---|---|
| 5 | 0.932 | -0.055 | -3.006 | 0.6697 |
| 10 | 0.713 | 0.348 | -3.136 | 0.7040 |
| 15 | 0.578 | 0.135 | -3.220 | 0.7127 |
| 20 | 0.462 | 0.121 | -3.246 | 0.7140 |
| 25 | 0.422 | 0.138 | -3.259 | 0.7137 |
| 30 | 0.398 | 0.223 | -3.266 | 0.7160 |
| 35 | 0.332 | 0.142 | -3.266 | 0.7133 |
| 40 | 0.351 | 0.037 | -3.275 | 0.7157 |
| 45 | 0.342 | 0.175 | -3.281 | 0.7240 |
| 50 | 0.349 | 0.179 | -3.275 | 0.7167 |
| 55 | 0.304 | 0.305 | -3.279 | 0.7203 |
| 60 | 0.309 | 0.152 | -3.284 | 0.7253 |
| 65 | 0.278 | 0.257 | -3.283 | 0.7250 |
| 70 | 0.309 | 0.361 | -3.284 | 0.7250 |
| 75 | 0.297 | 0.346 | -3.284 | 0.7247 |
| 80 | 0.284 | 0.390 | -3.286 | 0.7263 |
| 85 | 0.304 | 0.383 | -3.289 | 0.7277 |
| 90 | 0.269 | 0.385 | -3.287 | 0.7277 |
| 95 | 0.327 | 0.395 | -3.286 | 0.7297 |
| 100 | 0.276 | 0.354 | -3.286 | 0.7287 |