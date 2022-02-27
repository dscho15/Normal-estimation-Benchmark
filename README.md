# Normal-estimation Benchmark
Testing: Cilantro / PCL (Normal / Omp)

# Findings

The findings are stated below; the mean is determined for 1000 iterations of normal estimation, with the same configuration, using different methods. There is a significant difference, between (pcl) and (cilantro/pcl_omp).

<p align="center">
    <img src="data/results.png" alt="Kitten" title="A cute kitten" />
</p>

The two last methods show similar performance, i.e., there is no significant difference between pcl using omp methods and cilantro, when estimating normals. The following configuration was used in all three experiments (knn, k=7).
