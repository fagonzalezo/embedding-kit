# E10: the default selection (WP 8.2, WP 8.3)

64 sets. Test stage, one seed. Primary outcome: test cosine 10-NN accuracy (kNN R^2 on regression sets); intervals: paired bootstrap over test points (2,000 resamples, 95%). Criteria: `docs/IMPLEMENTATION_PLAN.md`, "WP 8.2 design and criteria".

## WP 8.2 verdicts (primary outcome)

- cap 5,000: binds on 47 sets; significantly worse on 6 (cifar100_dinov2 -2.3, cifar10_vit -1.7, emotion_bge -3.5, emotion_e5 -6.2, emotion_gte -3.5, emotion_minilm -3.6); same choice on 38 of 47. **FAIL**
- cap 10,000: binds on 40 sets; significantly worse on 0; same choice on 39 of 40. **pass**
- cap 20,000: binds on 11 sets; significantly worse on 0; same choice on 11 of 11. **pass**
- default cap 50,000: binds on no set (no E4 set has more than 50,000 training rows)

### Cap 5,000 against selection on all rows

| set | choice | reference choice | rows used | diff (points) | 95% interval |
|---|---|---|---|---|---|
| agnews_bge | identity | identity | 5000 | +0.00 | [+0.00, +0.00] |
| agnews_distilroberta | identity | identity | 5000 | +0.00 | [+0.00, +0.00] |
| agnews_e5 | identity | identity | 5000 | +0.00 | [+0.00, +0.00] |
| agnews_glove | identity | identity | 5000 | +0.00 | [+0.00, +0.00] |
| agnews_gte | pca-64 | identity | 5000 | +0.00 | [-0.54, +0.52] |
| agnews_minilm | identity | identity | 5000 | +0.00 | [+0.00, +0.00] |
| agnews_mpnet | identity | identity | 5000 | +0.00 | [+0.00, +0.00] |
| banking77_bge | refiner-128 | identity | 5000 | -0.03 | [-0.72, +0.65] |
| banking77_distilroberta | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| banking77_e5 | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| banking77_glove | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| banking77_gte | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| banking77_minilm | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| banking77_mpnet | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| cifar100_clip | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| cifar100_convnext | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| cifar100_dinov2 | identity | refiner-128 | 5000 | -2.31 **worse** | [-2.82, -1.77] |
| cifar100_resnet50 | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| cifar100_vit | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| cifar10_clip | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| cifar10_convnext | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| cifar10_dinov2 | identity | refiner-128 | 5000 | -0.19 | [-0.40, +0.02] |
| cifar10_resnet50 | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| cifar10_vit | identity | refiner-128 | 5000 | -1.73 **worse** | [-2.14, -1.29] |
| emotion_bge | identity | refiner-128 | 5000 | -3.50 **worse** | [-5.45, -1.50] |
| emotion_distilroberta | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| emotion_e5 | identity | refiner-128 | 5000 | -6.20 **worse** | [-8.25, -4.15] |
| emotion_glove | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| emotion_gte | identity | refiner-128 | 5000 | -3.45 **worse** | [-5.55, -1.45] |
| emotion_minilm | identity | refiner-128 | 5000 | -3.65 **worse** | [-5.70, -1.60] |
| emotion_mpnet | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| fluorescence_esm2 | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| massive_bge | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| massive_distilroberta | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| massive_e5 | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| massive_glove | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| massive_gte | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| massive_minilm | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| massive_mpnet | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| newsgroups_bge | identity | identity | 5000 | +0.00 | [+0.00, +0.00] |
| newsgroups_distilroberta | identity | identity | 5000 | +0.00 | [+0.00, +0.00] |
| newsgroups_e5 | identity | identity | 5000 | +0.00 | [+0.00, +0.00] |
| newsgroups_glove | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |
| newsgroups_gte | identity | identity | 5000 | +0.00 | [+0.00, +0.00] |
| newsgroups_minilm | identity | identity | 5000 | +0.00 | [+0.00, +0.00] |
| newsgroups_mpnet | identity | identity | 5000 | +0.00 | [+0.00, +0.00] |
| yelp_mpnet | refiner-128 | refiner-128 | 5000 | +0.00 | [+0.00, +0.00] |

### Cap 10,000 against selection on all rows

| set | choice | reference choice | rows used | diff (points) | 95% interval |
|---|---|---|---|---|---|
| agnews_bge | identity | identity | 10000 | +0.00 | [+0.00, +0.00] |
| agnews_distilroberta | identity | identity | 10000 | +0.00 | [+0.00, +0.00] |
| agnews_e5 | identity | identity | 10000 | +0.00 | [+0.00, +0.00] |
| agnews_glove | identity | identity | 10000 | +0.00 | [+0.00, +0.00] |
| agnews_gte | identity | identity | 10000 | +0.00 | [+0.00, +0.00] |
| agnews_minilm | identity | identity | 10000 | +0.00 | [+0.00, +0.00] |
| agnews_mpnet | identity | identity | 10000 | +0.00 | [+0.00, +0.00] |
| cifar100_clip | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| cifar100_convnext | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| cifar100_dinov2 | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| cifar100_resnet50 | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| cifar100_vit | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| cifar10_clip | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| cifar10_convnext | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| cifar10_dinov2 | identity | refiner-128 | 10000 | -0.19 | [-0.40, +0.02] |
| cifar10_resnet50 | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| cifar10_vit | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| emotion_bge | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| emotion_distilroberta | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| emotion_e5 | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| emotion_glove | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| emotion_gte | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| emotion_minilm | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| emotion_mpnet | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| fluorescence_esm2 | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| massive_bge | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| massive_distilroberta | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| massive_e5 | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| massive_glove | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| massive_gte | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| massive_minilm | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| massive_mpnet | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| newsgroups_bge | identity | identity | 10000 | +0.00 | [+0.00, +0.00] |
| newsgroups_distilroberta | identity | identity | 10000 | +0.00 | [+0.00, +0.00] |
| newsgroups_e5 | identity | identity | 10000 | +0.00 | [+0.00, +0.00] |
| newsgroups_glove | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |
| newsgroups_gte | identity | identity | 10000 | +0.00 | [+0.00, +0.00] |
| newsgroups_minilm | identity | identity | 10000 | +0.00 | [+0.00, +0.00] |
| newsgroups_mpnet | identity | identity | 10000 | +0.00 | [+0.00, +0.00] |
| yelp_mpnet | refiner-128 | refiner-128 | 10000 | +0.00 | [+0.00, +0.00] |

### Cap 20,000 against selection on all rows

| set | choice | reference choice | rows used | diff (points) | 95% interval |
|---|---|---|---|---|---|
| cifar100_clip | refiner-128 | refiner-128 | 20000 | +0.00 | [+0.00, +0.00] |
| cifar100_convnext | refiner-128 | refiner-128 | 20000 | +0.00 | [+0.00, +0.00] |
| cifar100_dinov2 | refiner-128 | refiner-128 | 20000 | +0.00 | [+0.00, +0.00] |
| cifar100_resnet50 | refiner-128 | refiner-128 | 20000 | +0.00 | [+0.00, +0.00] |
| cifar100_vit | refiner-128 | refiner-128 | 20000 | +0.00 | [+0.00, +0.00] |
| cifar10_clip | refiner-128 | refiner-128 | 20000 | +0.00 | [+0.00, +0.00] |
| cifar10_convnext | refiner-128 | refiner-128 | 20000 | +0.00 | [+0.00, +0.00] |
| cifar10_dinov2 | refiner-128 | refiner-128 | 20000 | +0.00 | [+0.00, +0.00] |
| cifar10_resnet50 | refiner-128 | refiner-128 | 20000 | +0.00 | [+0.00, +0.00] |
| cifar10_vit | refiner-128 | refiner-128 | 20000 | +0.00 | [+0.00, +0.00] |
| fluorescence_esm2 | refiner-128 | refiner-128 | 20000 | +0.00 | [+0.00, +0.00] |

### Default cap 50,000 against selection on all rows

(none)

## Secondary outcomes (count of sets significantly worse / better)

| comparison | metric | sets | worse | better |
|---|---|---|---|---|
| cap10000 | knn_accuracy_euclidean | 38 | 1 | 0 |
| cap10000 | linear_probe_c1 | 38 | 1 | 0 |
| cap20000 | knn_accuracy_euclidean | 10 | 0 | 0 |
| cap20000 | linear_probe_c1 | 10 | 0 | 0 |
| cap5000 | knn_accuracy_euclidean | 45 | 7 | 0 |
| cap5000 | linear_probe_c1 | 45 | 6 | 0 |
| vs_identity | knn_accuracy_euclidean | 60 | 0 | 41 |
| vs_identity | linear_probe_c1 | 60 | 5 | 21 |

## WP 8.3: the default against the input

Primary outcome: test cosine 10-NN accuracy (classification) or kNN R^2 (regression); differences in points (x 100).

- E4: 35 sets; significantly worse on 0; significantly better on 21
- E4X: 25 sets; significantly worse on 0; significantly better on 18
- regression: 4 sets; significantly worse on 0; significantly better on 4
- **pass**: the default is significantly worse than the input on 0 of 64 sets

| set | choice | reference choice | rows used | diff (points) | 95% interval |
|---|---|---|---|---|---|
| agnews_bge | identity | identity | 20000 | +0.00 | [+0.00, +0.00] |
| agnews_distilroberta | identity | identity | 20000 | +0.00 | [+0.00, +0.00] |
| agnews_e5 | identity | identity | 20000 | +0.00 | [+0.00, +0.00] |
| agnews_glove | identity | identity | 20000 | +0.00 | [+0.00, +0.00] |
| agnews_gte | identity | identity | 20000 | +0.00 | [+0.00, +0.00] |
| agnews_minilm | identity | identity | 20000 | +0.00 | [+0.00, +0.00] |
| agnews_mpnet | identity | identity | 20000 | +0.00 | [+0.00, +0.00] |
| banking77_bge | identity | identity | 9993 | +0.00 | [+0.00, +0.00] |
| banking77_distilroberta | refiner-128 | identity | 9993 | +1.53 better | [+0.75, +2.31] |
| banking77_e5 | refiner-128 | identity | 9993 | +1.89 better | [+1.07, +2.70] |
| banking77_glove | refiner-128 | identity | 9993 | +5.27 better | [+4.13, +6.40] |
| banking77_gte | refiner-128 | identity | 9993 | +0.85 better | [+0.10, +1.63] |
| banking77_minilm | refiner-128 | identity | 9993 | +1.24 better | [+0.49, +2.02] |
| banking77_mpnet | refiner-128 | identity | 9993 | +0.81 better | [+0.07, +1.56] |
| cifar100_clip | refiner-128 | identity | 50000 | +8.36 better | [+7.57, +9.15] |
| cifar100_convnext | refiner-128 | identity | 50000 | +7.41 better | [+6.62, +8.24] |
| cifar100_dinov2 | refiner-128 | identity | 50000 | +2.31 better | [+1.77, +2.82] |
| cifar100_resnet50 | refiner-128 | identity | 50000 | +9.07 better | [+8.17, +9.95] |
| cifar100_vit | refiner-128 | identity | 50000 | +6.20 better | [+5.44, +6.95] |
| cifar10_clip | refiner-128 | identity | 50000 | +2.46 better | [+2.00, +2.94] |
| cifar10_convnext | refiner-128 | identity | 50000 | +2.83 better | [+2.37, +3.32] |
| cifar10_dinov2 | refiner-128 | identity | 50000 | +0.19 | [-0.02, +0.40] |
| cifar10_resnet50 | refiner-128 | identity | 50000 | +3.99 better | [+3.36, +4.65] |
| cifar10_vit | refiner-128 | identity | 50000 | +1.73 better | [+1.29, +2.14] |
| emotion_bge | refiner-128 | identity | 16000 | +3.50 better | [+1.50, +5.45] |
| emotion_distilroberta | refiner-128 | identity | 16000 | +7.20 better | [+5.00, +9.30] |
| emotion_e5 | refiner-128 | identity | 16000 | +6.20 better | [+4.15, +8.25] |
| emotion_glove | refiner-128 | identity | 16000 | +14.55 better | [+12.30, +16.80] |
| emotion_gte | refiner-128 | identity | 16000 | +3.45 better | [+1.45, +5.55] |
| emotion_minilm | refiner-128 | identity | 16000 | +3.65 better | [+1.60, +5.70] |
| emotion_mpnet | refiner-128 | identity | 16000 | +5.90 better | [+3.80, +8.10] |
| flowers102_clip | refiner-128 | identity | 2040 | +4.15 better | [+3.50, +4.81] |
| flowers102_convnext | refiner-128 | identity | 2040 | +10.91 better | [+10.00, +11.79] |
| flowers102_dinov2 | identity | identity | 2040 | +0.00 | [+0.00, +0.00] |
| flowers102_resnet50 | refiner-128 | identity | 2040 | +15.03 better | [+13.95, +16.07] |
| flowers102_vit | refiner-128 | identity | 2040 | +15.12 better | [+14.05, +16.13] |
| fluorescence_esm2 | refiner-128 | identity | 21446 | +127.48 better | [+125.44, +129.58] |
| massive_bge | refiner-128 | identity | 11514 | +4.24 better | [+3.16, +5.41] |
| massive_distilroberta | refiner-128 | identity | 11514 | +5.68 better | [+4.47, +6.86] |
| massive_e5 | refiner-128 | identity | 11514 | +5.99 better | [+4.74, +7.26] |
| massive_glove | refiner-128 | identity | 11514 | +6.32 better | [+4.98, +7.73] |
| massive_gte | refiner-128 | identity | 11514 | +7.70 better | [+6.46, +8.98] |
| massive_minilm | refiner-128 | identity | 11514 | +5.99 better | [+4.74, +7.30] |
| massive_mpnet | refiner-128 | identity | 11514 | +4.30 better | [+3.06, +5.55] |
| newsgroups_bge | identity | identity | 11314 | +0.00 | [+0.00, +0.00] |
| newsgroups_distilroberta | identity | identity | 11314 | +0.00 | [+0.00, +0.00] |
| newsgroups_e5 | identity | identity | 11314 | +0.00 | [+0.00, +0.00] |
| newsgroups_glove | refiner-128 | identity | 11314 | +5.47 better | [+4.42, +6.52] |
| newsgroups_gte | identity | identity | 11314 | +0.00 | [+0.00, +0.00] |
| newsgroups_minilm | identity | identity | 11314 | +0.00 | [+0.00, +0.00] |
| newsgroups_mpnet | identity | identity | 11314 | +0.00 | [+0.00, +0.00] |
| pets_clip | refiner-128 | identity | 3680 | +13.49 better | [+12.21, +14.80] |
| pets_convnext | refiner-128 | identity | 3680 | +2.02 better | [+1.20, +2.81] |
| pets_dinov2 | refiner-128 | identity | 3680 | +0.93 better | [+0.27, +1.61] |
| pets_resnet50 | refiner-128 | identity | 3680 | +1.80 better | [+1.09, +2.51] |
| pets_vit | refiner-128 | identity | 3680 | +1.36 better | [+0.60, +2.18] |
| stl10_clip | identity | identity | 5000 | +0.00 | [+0.00, +0.00] |
| stl10_convnext | identity | identity | 5000 | +0.00 | [+0.00, +0.00] |
| stl10_dinov2 | identity | identity | 5000 | +0.00 | [+0.00, +0.00] |
| stl10_resnet50 | identity | identity | 5000 | +0.00 | [+0.00, +0.00] |
| stl10_vit | pca-256 | identity | 5000 | -0.07 | [-0.22, +0.07] |
| synthetic_regression | refiner-128 | identity | 3000 | +50.61 better | [+46.20, +55.59] |
| synthetic_regression_ties | refiner-128 | identity | 3000 | +45.92 better | [+43.62, +48.34] |
| yelp_mpnet | refiner-128 | identity | 20000 | +16.91 better | [+15.13, +18.79] |

## WP 8.3: chosen candidates and the best in hindsight

- E4: `refiner-128` 22, `identity` 13; regret 0 on 26 of 35, largest 1.18 points
- E4X: `refiner-128` 18, `identity` 6, `pca-256` 1; regret 0 on 20 of 25, largest 0.60 points
- regression: `refiner-128` 4; regret 0 on 4 of 4, largest 0.00 points

| set | chosen | select | identity | best in hindsight | best | regret (points) |
|---|---|---|---|---|---|---|
| agnews_bge | identity | 0.9142 | 0.9142 | center_l2 | 0.9168 | 0.26 |
| agnews_distilroberta | identity | 0.9040 | 0.9040 | refiner | 0.9052 | 0.12 |
| agnews_e5 | identity | 0.9138 | 0.9138 | pca_l2-128 | 0.9172 | 0.34 |
| agnews_glove | identity | 0.8988 | 0.8988 | identity | 0.8988 | 0.00 |
| agnews_gte | identity | 0.9146 | 0.9146 | pca_l2-128 | 0.9176 | 0.30 |
| agnews_minilm | identity | 0.9062 | 0.9062 | identity | 0.9062 | 0.00 |
| agnews_mpnet | identity | 0.9094 | 0.9094 | center_l2 | 0.9112 | 0.18 |
| banking77_bge | identity | 0.9373 | 0.9373 | identity | 0.9373 | 0.00 |
| banking77_distilroberta | refiner-128 | 0.9360 | 0.9207 | refiner | 0.9360 | 0.00 |
| banking77_e5 | refiner-128 | 0.9369 | 0.9181 | refiner | 0.9369 | 0.00 |
| banking77_glove | refiner-128 | 0.8641 | 0.8114 | refiner | 0.8641 | 0.00 |
| banking77_gte | refiner-128 | 0.9376 | 0.9291 | refiner | 0.9376 | 0.00 |
| banking77_minilm | refiner-128 | 0.9314 | 0.9191 | refiner | 0.9314 | 0.00 |
| banking77_mpnet | refiner-128 | 0.9395 | 0.9314 | refiner | 0.9395 | 0.00 |
| cifar100_clip | refiner-128 | 0.7811 | 0.6975 | refiner | 0.7811 | 0.00 |
| cifar100_convnext | refiner-128 | 0.7852 | 0.7111 | refiner | 0.7852 | 0.00 |
| cifar100_dinov2 | refiner-128 | 0.9023 | 0.8792 | refiner | 0.9023 | 0.00 |
| cifar100_resnet50 | refiner-128 | 0.7150 | 0.6243 | refiner | 0.7150 | 0.00 |
| cifar100_vit | refiner-128 | 0.8016 | 0.7396 | refiner | 0.8016 | 0.00 |
| cifar10_clip | refiner-128 | 0.9510 | 0.9264 | refiner | 0.9510 | 0.00 |
| cifar10_convnext | refiner-128 | 0.9468 | 0.9185 | refiner | 0.9468 | 0.00 |
| cifar10_dinov2 | refiner-128 | 0.9851 | 0.9832 | refiner | 0.9851 | 0.00 |
| cifar10_resnet50 | refiner-128 | 0.9085 | 0.8686 | refiner | 0.9085 | 0.00 |
| cifar10_vit | refiner-128 | 0.9545 | 0.9372 | refiner | 0.9545 | 0.00 |
| emotion_bge | refiner-128 | 0.7720 | 0.7370 | refiner | 0.7720 | 0.00 |
| emotion_distilroberta | refiner-128 | 0.7210 | 0.6490 | refiner | 0.7210 | 0.00 |
| emotion_e5 | refiner-128 | 0.7735 | 0.7115 | refiner | 0.7735 | 0.00 |
| emotion_glove | refiner-128 | 0.7505 | 0.6050 | refiner | 0.7505 | 0.00 |
| emotion_gte | refiner-128 | 0.7690 | 0.7345 | refiner | 0.7690 | 0.00 |
| emotion_minilm | refiner-128 | 0.7180 | 0.6815 | refiner | 0.7180 | 0.00 |
| emotion_mpnet | refiner-128 | 0.6815 | 0.6225 | refiner | 0.6815 | 0.00 |
| flowers102_clip | refiner-128 | 0.9397 | 0.8982 | refiner | 0.9397 | 0.00 |
| flowers102_convnext | refiner-128 | 0.9008 | 0.7917 | refiner | 0.9008 | 0.00 |
| flowers102_dinov2 | identity | 0.9963 | 0.9963 | refiner | 0.9969 | 0.07 |
| flowers102_resnet50 | refiner-128 | 0.8252 | 0.6749 | refiner | 0.8252 | 0.00 |
| flowers102_vit | refiner-128 | 0.8671 | 0.7159 | refiner | 0.8671 | 0.00 |
| fluorescence_esm2 | refiner-128 | 0.4811 | -0.7938 | refiner | 0.4811 | 0.00 |
| massive_bge | refiner-128 | 0.8837 | 0.8413 | refiner | 0.8837 | 0.00 |
| massive_distilroberta | refiner-128 | 0.8709 | 0.8141 | refiner | 0.8709 | 0.00 |
| massive_e5 | refiner-128 | 0.8853 | 0.8255 | refiner | 0.8853 | 0.00 |
| massive_glove | refiner-128 | 0.7858 | 0.7226 | refiner | 0.7858 | 0.00 |
| massive_gte | refiner-128 | 0.8779 | 0.8009 | refiner | 0.8779 | 0.00 |
| massive_minilm | refiner-128 | 0.8601 | 0.8003 | refiner | 0.8601 | 0.00 |
| massive_mpnet | refiner-128 | 0.8662 | 0.8231 | refiner | 0.8662 | 0.00 |
| newsgroups_bge | identity | 0.6981 | 0.6981 | center_l2 | 0.7069 | 0.88 |
| newsgroups_distilroberta | identity | 0.6697 | 0.6697 | refiner | 0.6757 | 0.60 |
| newsgroups_e5 | identity | 0.6918 | 0.6918 | refiner | 0.6978 | 0.60 |
| newsgroups_glove | refiner-128 | 0.6078 | 0.5531 | refiner | 0.6078 | 0.00 |
| newsgroups_gte | identity | 0.7096 | 0.7096 | identity | 0.7096 | 0.00 |
| newsgroups_minilm | identity | 0.6738 | 0.6738 | center_l2 | 0.6771 | 0.33 |
| newsgroups_mpnet | identity | 0.7033 | 0.7033 | refiner | 0.7151 | 1.18 |
| pets_clip | refiner-128 | 0.8907 | 0.7558 | refiner | 0.8907 | 0.00 |
| pets_convnext | refiner-128 | 0.9297 | 0.9095 | refiner | 0.9297 | 0.00 |
| pets_dinov2 | refiner-128 | 0.9534 | 0.9441 | refiner | 0.9534 | 0.00 |
| pets_resnet50 | refiner-128 | 0.9193 | 0.9013 | refiner | 0.9193 | 0.00 |
| pets_vit | refiner-128 | 0.9272 | 0.9136 | refiner | 0.9272 | 0.00 |
| stl10_clip | identity | 0.9825 | 0.9825 | refiner | 0.9865 | 0.40 |
| stl10_convnext | identity | 0.9841 | 0.9841 | center_l2 | 0.9844 | 0.02 |
| stl10_dinov2 | identity | 0.9961 | 0.9961 | identity | 0.9961 | 0.00 |
| stl10_resnet50 | identity | 0.9768 | 0.9768 | identity | 0.9768 | 0.00 |
| stl10_vit | pca-256 | 0.9824 | 0.9831 | identity | 0.9831 | 0.08 |
| synthetic_regression | refiner-128 | 0.8889 | 0.3829 | refiner | 0.8889 | 0.00 |
| synthetic_regression_ties | refiner-128 | 0.9011 | 0.4419 | refiner | 0.9011 | 0.00 |
| yelp_mpnet | refiner-128 | 0.6883 | 0.5192 | refiner | 0.6883 | 0.00 |
