# CSE/ISyE 6740 - ML : Homework Reports – Georgia Tech CS
This repository contains the final reports for all homework assignments and final group project completed in **CSE 6740 / ISyE 6740: Computational Data Analy**, as part of the Georgia Tech Computer Science program. Only the written reports are public; source code and datasets remain private.

---
## Homework Reports

| Homework | Report |
|----------|--------|
| HW1      | [zliu943_HW1_report.pdf](HW1/zliu943_HW1_report.pdf) |
| HW2      | [zliu943_HW2_report.pdf](HW2/zliu943_HW2_report.pdf) |
| HW3      | [zliu943_HW3_report.pdf](HW3/zliu943_HW3_report.pdf) |
| HW4      | [zliu943_HW4_report.pdf](HW4/zliu943_HW4_report.pdf) |
| HW5      | [zliu943_HW5_report.pdf](HW5/zliu943_HW5_report.pdf) |
| HW6      | [zliu943_HW6_report.pdf](HW6/zliu943_HW6_report.pdf) |
| Final Project | [Group_Project_Report.pdf](Group%20Project/Group_Project_Report.pdf) |


---
## Homework Highlights

### HW 1  · Image Compression with K‑means
* Implemented K‑means from scratch (squared‑Euclidean metric)
* Compressed multiple RGB images with *k* = 2–32 clusters; elbow method for optimal *k*
* Analyzed convergence speed and runtime

### HW 2  · PCA & ISOMAP
* PCA on European food‑consumption (16 countries × 20 foods); visualized country clusters
* Built eigenfaces recognizer (YaleFaces) via PCA
* Implemented ε‑ISOMAP on 698 face images; 2‑D manifold reveals head‑pose variation

### HW 3  · Density Estimation & EM‑GMM
* 1‑D / 2‑D KDE on amygdala–ACC dataset; conditional density & independence tests
* Wrote EM for Gaussian Mixture Models; clustered MNIST digits “2” vs “6”
* Visualized log‑likelihood convergence, mean digit images, 4×4 covariance heat‑maps

### HW 4  · Optimization & Small‑scale Spam / Divorce Classifiers
* Derived gradient & Hessian for 1‑D logistic regression; provided batch‑GD & SGD pseudocode
* Hand‑computed Naive Bayes spam filter; compared NB / LogReg / KNN on divorce data (94 % accuracy)

### HW 5  · Multi‑classifier Benchmark & AdaBoost
* Bench‑marked KNN, LogReg, linear‑SVM, RBF‑SVM, 2‑layer MLP on 10‑class MNIST; reported precision/recall/F₁
* Implemented 3‑round AdaBoost with decision stumps; achieved zero training error
* Reconstructed sparse medical images via Lasso vs Ridge; Lasso better on 52 % sparsity

### HW 6  · Ensemble Learning & Advanced Regression
* CART, Random Forest, One‑class SVM on UCI spam (RF best after tuning)
* 3‑round hand‑calculated AdaBoost stump demo (training error 0)
* Locally Weighted Linear Regression with Gaussian kernel; 5‑fold CV for bandwidth; illustrated bias‑variance trade‑off

---
## Final Group Project
**WiDS Datathon 2024 – Predicting Timely Treatment for Metastatic Triple-Negative Breast Cancer (TNBC)** 
We tackled the healthcare challenge of predicting whether patients with metastatic triple-negative breast cancer (TNBC) would receive a diagnosis within 90 days of their initial screening. Our analysis integrates patient demographics, socioeconomic indicators, and environmental factors to model the likelihood of timely diagnosis.
[Official Competition Link](https://www.kaggle.com/competitions/widsdatathon2024-challenge1)

> The shapefile dataset (`tl_2020_us_zcta520.shp`, ~781MB) used in this project was too large to upload to GitHub.
To access the data, visit the official U.S. Census Bureau page:  
[https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)


---

## Author

Braxton Liu  
[https://github.com/braxtonliu](https://github.com/braxtonliu)  
