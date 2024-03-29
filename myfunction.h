#include <iostream>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/TextureMesh.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/obj_io.h>

#include <pcl/features/vfh.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/pfh.h>
#include <pcl/features/fpfh_omp.h>

#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/uniform_sampling.h>

#include <pcl/registration/transforms.h>
#include <pcl/registration/registration.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/correspondence_estimation.h>

#include <pcl/surface/texture_mapping.h>
#include <pcl/surface/impl/mls.hpp>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>


typedef pcl::PointXYZ PointType;
typedef pcl::FPFHSignature33 FeatureType;

typedef struct func {
	void (*fptr) ();
	int key;
} func;

const float DOWNSAMPLE_RATE = 0.005f; // big rate -> small points
const int KSEARCH_SIZE = 200;


void prtUsage();

void compute_surface_normals(pcl::PointCloud<PointType>::Ptr &points, pcl::PointCloud<pcl::Normal>::Ptr &normals);

void downsample(const pcl::PointCloud<PointType>::Ptr &points, pcl::PointCloud<PointType>::Ptr &points_downsample);

void visualize_normals(const pcl::PointCloud<PointType>::Ptr points, pcl::PointCloud<pcl::Normal>::Ptr normals);

void visualize_pointcloud(const pcl::PointCloud<PointType>::Ptr points);

void normals_demo();

void pointcloud_demo();

void vfh_demo();

void pointcloud_process();

void model_to_plane_point();