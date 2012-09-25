#include <iostream>
#include <vector>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_types.h>
#include <pcl/features/vfh.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/registration/transforms.h>
#include <pcl/registration/registration.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef pcl::PointXYZ PointType;
typedef pcl::FPFHSignature33 FeatureType;

typedef struct func {
	void (*fptr) ();
	int key;
} func;

const float DOWNSAMPLE_RATE = 0.05f; // big rate -> small points
const int KSEARCH_SIZE = 100;


void prtUsage();

void compute_surface_normals(pcl::PointCloud<PointType>::Ptr &points, pcl::PointCloud<pcl::Normal>::Ptr &normals);

void downsample(const pcl::PointCloud<PointType>::Ptr &points, pcl::PointCloud<PointType>::Ptr &points_downsample);

void visualize_normals(const pcl::PointCloud<PointType>::Ptr points, pcl::PointCloud<pcl::Normal>::Ptr normals);

void visualize_pointcloud(const pcl::PointCloud<PointType>::Ptr points);

void normals_demo();

void pointcloud_demo();

void vfh_demo();

void pointcloud_process();