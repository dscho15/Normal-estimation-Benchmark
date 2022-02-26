#include <iostream>
#include <iomanip>

#include <tqdm.hpp>

#include <cilantro/utilities/point_cloud.hpp>
#include <cilantro/visualization.hpp>
#include <cilantro/utilities/timer.hpp>

#include <pangolin/display/display.h>
#include <pangolin/plot/plotter.h>

#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/visualization/pcl_visualizer.h>

template<typename T>
T variance(const std::vector<T> &vec) {
    const size_t sz = vec.size();
    if (sz == 1) {
        return 0.0;
    }

    // Calculate the mean
    const T mean = std::accumulate(vec.begin(), vec.end(), 0.0) / sz;

    // Now calculate the variance
    auto variance_func = [&mean, &sz](T accumulator, const T& val) {
        return accumulator + ((val - mean)*(val - mean) / (sz - 1));
    };

    return std::accumulate(vec.begin(), vec.end(), 0.0, variance_func);
}

std::vector<double> experiment_cilantro(const int & iterations, const std::string & path);
std::vector<double> experiment_pcl(const int & iterations, const std::string & path);
std::vector<double> experiment_pcl_omp(const int & iterations, const std::string & path);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr filter_voxel_grid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & inputcloud, const float & voxel_x, const float & voxel_y, const float & voxel_z);

int main(int argc, char const *argv[])
{
    // Path to cloud
    std::string path = "/home/daniel/Desktop/pointcloud_benchmark/data/frame_1.ply";
    int iterations = 1000;
    for (const auto& time : {experiment_pcl(iterations, path), experiment_cilantro(iterations, path), experiment_pcl_omp(iterations, path)})
    {
        std::cout << std::setw(50) << "The mean: " << std::accumulate(time.begin(), time.end(), 0.)/time.size() << "[s]" << std::endl;
        std::cout << std::setw(50) << "The standard deviation: " << sqrt(variance(time)) << "[s]" << std::endl;
    }
    return 0;
}

std::vector<double> experiment_cilantro(const int & iterations, const std::string & path)
{
    std::vector<double> time;
    cilantro::PointCloud3f cloud(path);
    tqdm bar;
    bar.set_label("cilantro");
    bar.set_theme_braille();

    if (cloud.isEmpty())
    {
        std::cout << "The input cloud is empty, shutting down" << std::endl;
        return time;
    }

    // Clear input normals
    cloud.normals.resize(Eigen::NoChange, 0);
    cloud.gridDownsample(0.005f);

    for(auto i = 0; i < iterations; ++i)
    {
        // tqdm-like
        bar.progress(i, iterations);

        // init timer
        cilantro::Timer tree_timer;
        tree_timer.start();
        cilantro::KDTree3f<> tree(cloud.points);
        tree_timer.stop();

        cilantro::Timer ne_timer;
        ne_timer.start();
        cloud.estimateNormalsKNN(tree, 7);
        ne_timer.stop();

        // elapsed time
        time.push_back(tree_timer.getElapsedTime() + ne_timer.getElapsedTime());
    }
    bar.finish();
    return time;

}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr filter_voxel_grid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & inputcloud, const float & voxel_x, const float & voxel_y, const float & voxel_z)
{
    pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2());
    pcl::toPCLPointCloud2(*inputcloud, *cloud);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr outputcloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::PCLPointCloud2::Ptr cloud_filtered(new pcl::PCLPointCloud2());

    // Create the filtering object
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(voxel_x, voxel_y, voxel_z);
    sor.filter(*cloud_filtered);

    pcl::fromPCLPointCloud2(*cloud_filtered, *outputcloud);

    return outputcloud;
}

std::vector<double> experiment_pcl(const int & iterations, const std::string & path)
{
    std::vector<double> time;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    tqdm bar;
    bar.set_label("pcl");
    bar.set_theme_braille();

    if (pcl::io::loadPLYFile(path, *point_cloud_ptr))
    {
        std::cout << "The input cloud is empty, shutting down" << std::endl;
        return time;
    }
    
    point_cloud_ptr = filter_voxel_grid(point_cloud_ptr, 0.005, 0.005, 0.005);

    for(auto i = 0; i < iterations; ++i)
    {
        // tqdm-like
        bar.progress(i, iterations);

        // init timer
        cilantro::Timer timer_;
        timer_.start();
        pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
        ne.setInputCloud(point_cloud_ptr);
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>());
        ne.setSearchMethod(tree);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals1 (new pcl::PointCloud<pcl::Normal>);
        ne.setKSearch(5);
        ne.compute(*cloud_normals1);
        timer_.stop();

        // elapsed time
        time.push_back(timer_.getElapsedTime());
    }
    bar.finish();

    return time;
}

std::vector<double> experiment_pcl_omp(const int & iterations, const std::string & path)
{
    std::vector<double> time;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    tqdm bar;
    bar.set_label("pcl_omp");
    bar.set_theme_braille();

    if (pcl::io::loadPLYFile(path, *point_cloud_ptr))
    {
        std::cout << "The input cloud is empty, shutting down" << std::endl;
        return time;
    }
    
    point_cloud_ptr = filter_voxel_grid(point_cloud_ptr, 0.005, 0.005, 0.005);

    for(auto i = 0; i < iterations; ++i)
    {
        // tqdm-like
        bar.progress(i, iterations);

        // init timer
        cilantro::Timer timer_;
        timer_.start();
        pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
        ne.setInputCloud(point_cloud_ptr);
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>());
        ne.setSearchMethod(tree);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals1(new pcl::PointCloud<pcl::Normal>);
        ne.setKSearch(5);
        ne.compute(*cloud_normals1);
        timer_.stop();

        // elapsed time
        time.push_back(timer_.getElapsedTime());
    }
    bar.finish();

    return time;
}