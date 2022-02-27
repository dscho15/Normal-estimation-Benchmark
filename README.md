# Normal-estimation Benchmark
Testing: Cilantro / PCL (Normal / Omp)

# Findings

The results; the mean is determined for 1000 iterations of normal estimation with the same configuration using different methods.

<p align="center">
    <img src="data/results.png" alt="Kitten" title="A cute kitten" />
</p>

The two last methods show similar performance, i.e., there is no significant difference between pcl using omp methods and cilantro, when estimating normals (knn, k=7).
