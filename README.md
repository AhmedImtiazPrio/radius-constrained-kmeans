# ROBUST K-MEANS VIA RADIUS UPPER BOUNDS
#### Codes for _"NO MORE THAN 6FT APART: ROBUST K-MEANS VIA RADIUS UPPER BOUNDS"_, ICASSP 2022 [[Paper Link](https://people.bengali.ai/wp-content/uploads/2021/11/Robust_Kmeans_ICASSP.pdf)]
Authors: Ahmed Imtiaz Humayun<sup>1</sup>, Randall Balestriero<sup>2</sup>, Anastasios Kyrillidis<sup>1</sup>, Richard Baraniuk<sup>1</sup>

<sup>1</sup>Rice University, <sup>2</sup>Meta AI Research

Abstract: _Centroid based clustering methods such as k-means, k-medoids and k-centers are heavily applied as a go-to tool in exploratory data analysis. In many cases, those methods are used to obtain representative centroids of the data manifold for visualization or summarization of a dataset. Real world datasets often contain inherent abnormalities, e.g., repeated samples and sampling bias, that manifest imbalanced clustering. We propose to remedy such a scenario by introducing a maximal radius constraint r on the clusters formed by the centroids, i.e., samples from the same cluster should not be more than 2r apart in terms of l2 distance. We achieve this constraint by solving a semi-definite program, followed by a linear assignment problem with quadratic constraints. Through qualitative results, we show that our proposed method is robust towards dataset imbalances and sampling artifacts. To the best of our knowledge, ours is the first constrained k-means clustering method with hard radius constraints._






## Requirements:
```
## For r-KMeans
cvxpy>=1.1.5
gurobi>=9.1.1 
mosek # can also be used with cvxpy.scs 
scikit-learn

## For KMedoids and cardinality-KMeans
scikit-learn-extra
k-means-constrained
```
## Usage:
Run the `two_moons.ipynb` to reproduce results from the paper.

## Cite:
```
@inproceedings{humayun2022rk,
  title={No more than 6ft Apart: Robust K-Means via Radius Upper Bounds},
  author={Humayun, Ahmed Imtiaz and Balestriero, Randall and Kyrillidis, Anastasios and Baraniuk, Richard},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2022},
  organization={IEEE}
}
```
