#include "myfunction.h"
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>

static string filename_1, filename_2;

void remove_nan()
{
	cout << "Input removed filename: ";
	cin >> filename_1;

	pcl::PointCloud<PointType>::Ptr points(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr points_out(new pcl::PointCloud<PointType>);

	pcl::io::loadPCDFile(filename_1, *points);

	points_out->width = points->points.size();
	points_out->height = 1;
	points_out->resize(points_out->width);
	int cnt = 0;
	for (size_t i = 0; i < points->points.size(); ++i) {
		if (pcl::isFinite(points->points[i])) {
			points_out->points[cnt] = points->points[i];
			++cnt;
		}
	}
	points_out->width = cnt;
	points_out->resize(points_out->width);
	points_out->is_dense = true;
	pcl::io::savePCDFile("output.pcd", *points_out);
}

/*
void detect_keypoints(pcl::PointCloud<PointType>::Ptr &points, float min_scale, int nr_octaves,
	int nr_scales_per_octave, float min_contrast, pcl::PointCloud<pcl::PointWithScale>::Ptr &keypoints)
{
	pcl::SIFTKeypoint<PointType, pcl::PointWithScale> sift;
	
	// Use a FLANN-based KdTree to perform neighborhood searches
	pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
	sift.setSearchMethod(tree);

	// Set the detection parameters
	sift.setScales(min_scale, nr_octaves, nr_scales_per_octave);
	sift.setMinimumContrast(min_contrast);

	// Set the input
	sift.setSearchSurface(points);
	sift.setInputCloud(points);

	// Detect the keypoints and store in keypoints
	sift.compute(*keypoints);

	std::cout << "SIFT find " << keypoints->size() << " keypoints." << std::endl;
}
*/

void visualize_keypoints(const pcl::PointCloud<PointType>::Ptr points,
	const pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints)
{
	// Add points to the visualizer
	pcl::visualization::PCLVisualizer viewer;
	viewer.addPointCloud(points, "points");

	// Draw each keypoint as a sphere
	for (size_t i = 0; i < keypoints->size(); ++i) {
		// Get the point data
		const pcl::PointWithScale &p = keypoints->points[i];

		// Generate a unique string for each sphere
		std::stringstream ss("keypoint");
		ss << i;

		//std::cout << p.scale << std::endl;
		// Add a sphere at the keypoint
		viewer.addSphere(p, p.scale, 1.0, 0.0, 0.0, ss.str());
	}

	// Give control over to the visualizer
	viewer.spin();
}

void keypoints_demo()
{
	cout << "Input filename: ";
	cin >> filename_1;

	// Create new point clouds to hold data
	pcl::PointCloud<PointType>::Ptr points (new pcl::PointCloud<PointType>);
	pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints (new pcl::PointCloud<pcl::PointWithScale>);

	// Load point cloud
	pcl::io::loadPCDFile(filename_1, *points);

	// Compute keypoints by SIFT
	const float min_scale = 0.002f;
	const int nr_octaves = 3;
	const int nr_octaves_per_scale = 5;
	const float min_contrast = 1.0f;
	//detect_keypoints(points, min_scale, nr_octaves, nr_octaves_per_scale, min_contrast, keypoints);

	// Visualize the point cloud and its keypoints
	visualize_keypoints(points, keypoints);
}

void compute_features_at_keypoints(pcl::PointCloud<PointType>::Ptr &points, pcl::PointCloud<pcl::Normal>::Ptr &normals,
	pcl::PointCloud<PointType>::Ptr &keypoints, pcl::PointCloud<FeatureType>::Ptr &descriptor)
{
	// Create FPFHEstimationOMP object
	pcl::FPFHEstimationOMP<PointType, pcl::Normal, FeatureType> fpfh;

	// Use a FLANN-based KdTree to perform neighborhood searches
	//pcl::search::KdTree<pcl::PointType>::Ptr tree (new pcl::search::KdTree<pcl::PointType>);
	fpfh.setKSearch(KSEARCH_SIZE);

	// Use all of the points to analyze
	fpfh.setSearchSurface(points);
	fpfh.setInputNormals(normals);

	// Only compute features at the keypoints
	fpfh.setInputCloud(keypoints);
	fpfh.compute(*descriptor);
}

void find_features_correspondences(pcl::PointCloud<FeatureType>::Ptr &source_descriptor,
	pcl::PointCloud<FeatureType>::Ptr &target_descriptor, pcl::Correspondences &correspondences)
{
	pcl::registration::CorrespondenceEstimation<FeatureType, FeatureType> corr_est;
	corr_est.setInputCloud(source_descriptor);
	corr_est.setInputTarget(target_descriptor);
	corr_est.determineCorrespondences(correspondences);
}

void visualize_correspondences(const pcl::PointCloud<PointType>::Ptr points1, const pcl::PointCloud<PointType>::Ptr keypoints1,
	const pcl::PointCloud<PointType>::Ptr points2, const pcl::PointCloud<PointType>::Ptr keypoints2, pcl::Correspondences &correspondences)
{
	// Create new point clouds to hold transformed data
	pcl::PointCloud<PointType>::Ptr points_left (new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr keypoints_left (new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr points_right (new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr keypoints_right (new pcl::PointCloud<PointType>);

	// Shift fist cloud to the left, second cloud to the right
	const Eigen::Vector3f translate(0.1, 0.0, 0.0);
	const Eigen::Quaternionf no_rotation(0, 0, 0, 0);

	pcl::transformPointCloud(*points1, *points_left, -translate, no_rotation);
	pcl::transformPointCloud(*keypoints1, *keypoints_left, -translate, no_rotation);

	pcl::transformPointCloud(*points2, *points_right, translate, no_rotation);
	pcl::transformPointCloud(*keypoints2, *keypoints_right, translate, no_rotation);

	// Add clouds to vizualizer
	pcl::visualization::PCLVisualizer viewer;
	viewer.addPointCloud(points_left, "points_left");
	viewer.addPointCloud(points_right, "points_right");

	// Draw lines between the best corresponding points
	for (size_t i = 0; i < keypoints_left->size(); ++i) {

		// Get the pair of points
		const PointType &p_left = keypoints_left->points[i];
		const PointType &p_right = keypoints_right->points[correspondences[i].index_match];

		// Generate a unique string for each line
		std::stringstream ss("line");
		ss << i;

		// Draw the line
		viewer.addLine(p_left, p_right, 0, 1, 0, ss.str());
	}

	// Given control over to the visualizer
	viewer.spin();
}

void coorespondences_demo()
{
	cout << "Input first data filename: ";
	cin >> filename_1;
	cout << "Input second data filename: ";
	cin >> filename_2;

	// Create new point clouds to hold two data
	pcl::PointCloud<PointType>::Ptr points1 (new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr keypoints1 (new pcl::PointCloud<PointType>);
	pcl::PointCloud<pcl::Normal>::Ptr normals1 (new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<FeatureType>::Ptr descriptor1 (new pcl::PointCloud<FeatureType>);

	pcl::PointCloud<PointType>::Ptr points2 (new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr keypoints2 (new pcl::PointCloud<PointType>);
	pcl::PointCloud<pcl::Normal>::Ptr normals2 (new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<FeatureType>::Ptr descriptor2 (new pcl::PointCloud<FeatureType>);

	// Load point cloud
	pcl::io::loadPCDFile(filename_1, *points1);
	pcl::io::loadPCDFile(filename_2, *points2);

	// Compute surface normals
	compute_surface_normals(points1, normals1);
	compute_surface_normals(points2, normals2);

	// Downsample points as keypoints
	downsample(points1, keypoints1);
	downsample(points2, keypoints2);

	// Compute features by keypoints
	compute_features_at_keypoints(points1, normals1, keypoints1, descriptor1);
	compute_features_at_keypoints(points2, normals2, keypoints2, descriptor2);

	// Find features correspondences
	pcl::Correspondences correspondences;
	find_features_correspondences(descriptor1, descriptor2, correspondences);

	// Visualize two point clouds and their features correspondences
	visualize_correspondences(points1, keypoints1, points2, keypoints2, correspondences);
}

void icp_demo()
{
	cout << "Input test data filename: ";
	cin >> filename_1;
	cout << "Input model filename: ";
	cin >> filename_2;

	// Create new point clouds to hold two data
	pcl::PointCloud<PointType>::Ptr points1 (new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr keypoints1 (new pcl::PointCloud<PointType>);
	pcl::PointCloud<pcl::Normal>::Ptr normals1 (new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<FeatureType>::Ptr descriptor1 (new pcl::PointCloud<FeatureType>);

	pcl::PointCloud<PointType>::Ptr points2 (new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr keypoints2 (new pcl::PointCloud<PointType>);
	pcl::PointCloud<pcl::Normal>::Ptr normals2 (new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<FeatureType>::Ptr descriptor2 (new pcl::PointCloud<FeatureType>);

	// Load point cloud
	pcl::io::loadPCDFile(filename_1, *points1);
	pcl::io::loadPCDFile(filename_2, *points2);

	// Compute surface normals
	compute_surface_normals(points1, normals1);
	compute_surface_normals(points2, normals2);

	// Downsample points as keypoints
	downsample(points1, keypoints1);
	downsample(points2, keypoints2);
	//keypoints1 = points1;
	//keypoints2 = points2;

	// Compute features by keypoints
	compute_features_at_keypoints(points1, normals1, keypoints1, descriptor1);
	compute_features_at_keypoints(points2, normals2, keypoints2, descriptor2);

	// Find features correspondences
	//pcl::Correspondences correspondences;
	//find_features_correspondences(descriptor1, descriptor2, correspondences);

	/*
	//pcl::CorrespondencesConstPtr correspondences_ptr(&correspondences);
	pcl::CorrespondencesConstPtr correspondences_ptr (new pcl::Correspondences(correspondences));
	pcl::Correspondences correspondences_reject;
	pcl::registration::CorrespondenceRejectorSampleConsensus<PointType> sac;
	sac.setInputCloud(points1);
	sac.setTargetCloud(points1);
	sac.setInlierThreshold(0.2);
	sac.setMaxIterations(10);
	sac.setInputCorrespondences(correspondences_ptr);
	sac.getCorrespondences(correspondences_reject);
	Eigen::Matrix4f transformation = sac.getBestTransformation();

	cout << transformation << endl;
	*/

	// Initial Alignment
	///*
	pcl::SampleConsensusInitialAlignment<PointType, PointType, FeatureType> sac_ia;
	sac_ia.setMinSampleDistance(0.035);
	sac_ia.setMaxCorrespondenceDistance(0.01);
	sac_ia.setMaximumIterations(1000);

	sac_ia.setInputCloud(keypoints1);
	sac_ia.setSourceFeatures(descriptor1);

	sac_ia.setInputTarget(keypoints2);
	sac_ia.setTargetFeatures(descriptor2);

	pcl::PointCloud<PointType>::Ptr initial_output (new pcl::PointCloud<PointType>);
	sac_ia.align(*initial_output);
	Eigen::Matrix4f initial_transform = sac_ia.getFinalTransformation();

	//pcl::PointCloud<PointType>::Ptr final (new pcl::PointCloud<PointType>);
	//pcl::transformPointCloud (*points1, *final, initial_transform);
	//*final += *points2;
	//*/
	
	// Refine
	///*
	pcl::PointCloud<PointType>::Ptr point1_transformed (new pcl::PointCloud<PointType>);
	pcl::transformPointCloud (*points1, *point1_transformed, initial_transform);

	pcl::IterativeClosestPoint<PointType, PointType> icp;
	typedef pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal> PointToPlane;
	boost::shared_ptr<PointToPlane> point_to_plane(new PointToPlane);
	//icp.setTransformationEstimation(point_to_plane);

	icp.setInputCloud (point1_transformed);
	icp.setInputTarget (points2);
	icp.setMaximumIterations (500);
	icp.setMaxCorrespondenceDistance (0.1);
	icp.setRANSACOutlierRejectionThreshold (0.05);

	pcl::PointCloud<PointType>::Ptr registration_output2 (new pcl::PointCloud<PointType>);
	icp.align(*registration_output2);

	Eigen::Matrix4f final_transform = icp.getFinalTransformation() * initial_transform;

	pcl::PointCloud<PointType>::Ptr final (new pcl::PointCloud<PointType>);
	pcl::transformPointCloud (*points1, *final, final_transform);
	*final += *points2;
	//*/

	visualize_pointcloud(final);
	//pcl::io::savePCDFile("final.pcd", *final);
}

void point_to_plane_icp()
{
	cout << "Input data filename: ";
	cin >> filename_1;
	cout << "Input model filename: ";
	cin >> filename_2;

	// Create new point clouds to hold two data
	pcl::PointCloud<pcl::PointXYZ>::Ptr points1 (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Normal>::Ptr normals1 (new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<FeatureType>::Ptr descriptor1 (new pcl::PointCloud<FeatureType>);

	pcl::PointCloud<PointType>::Ptr points2 (new pcl::PointCloud<PointType>);
	pcl::PointCloud<pcl::Normal>::Ptr normals2 (new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<FeatureType>::Ptr descriptor2 (new pcl::PointCloud<FeatureType>);

	pcl::io::loadPCDFile(filename_1, *points1);
	pcl::io::loadPCDFile(filename_2, *points2);

	pcl::PointCloud<pcl::PointNormal>::Ptr src(new pcl::PointCloud<pcl::PointNormal>);
	pcl::copyPointCloud(*points1, *src);
	pcl::PointCloud<pcl::PointNormal>::Ptr tgt(new pcl::PointCloud<pcl::PointNormal>);
	pcl::copyPointCloud(*points2, *tgt);

	pcl::NormalEstimation<pcl::PointNormal, pcl::PointNormal> norm_est;
	norm_est.setKSearch(KSEARCH_SIZE);

	norm_est.setInputCloud(src);
	norm_est.compute(*src);

	norm_est.setInputCloud(tgt);
	norm_est.compute(*tgt);

	// Compute feature
	pcl::FPFHEstimationOMP<pcl::PointNormal, pcl::Normal, FeatureType> fpfh;

	fpfh.setKSearch(KSEARCH_SIZE);

	fpfh.setSearchSurface(src);
	pcl::copyPointCloud(*src, *normals1);
	fpfh.setInputNormals(normals1);
	fpfh.setInputCloud(src);
	fpfh.compute(*descriptor1);

	fpfh.setSearchSurface(tgt);
	pcl::copyPointCloud(*tgt, *normals2);
	fpfh.setInputNormals(normals2);
	fpfh.setInputCloud(tgt);
	fpfh.compute(*descriptor2);

	// Initial alignment
	pcl::SampleConsensusInitialAlignment<pcl::PointNormal, pcl::PointNormal, FeatureType> sac_ia;
	sac_ia.setMinSampleDistance(0.035);
	sac_ia.setMaxCorrespondenceDistance(0.01);
	sac_ia.setMaximumIterations(1000);

	sac_ia.setInputCloud(src);
	sac_ia.setSourceFeatures(descriptor1);

	sac_ia.setInputTarget(tgt);
	sac_ia.setTargetFeatures(descriptor2);

	pcl::PointCloud<pcl::PointNormal>::Ptr initial_output (new pcl::PointCloud<pcl::PointNormal>);
	sac_ia.align(*initial_output);
	Eigen::Matrix4f initial_transform = sac_ia.getFinalTransformation();


	pcl::transformPointCloud (*src, *src, initial_transform);
	pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> icp;
	typedef pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal> PointToPlane;
	boost::shared_ptr<PointToPlane> point_to_plane(new PointToPlane);
	icp.setTransformationEstimation(point_to_plane);
	icp.setInputCloud(src);
	icp.setInputTarget(tgt);
	icp.setRANSACOutlierRejectionThreshold (0.05);
	icp.setRANSACIterations(100);
	icp.setMaximumIterations(500);
	icp.setTransformationEpsilon(1e-3);

	pcl::PointCloud<pcl::PointNormal> result;
	icp.align(result);

	Eigen::Matrix4f final_transform = icp.getFinalTransformation() * initial_transform;

	pcl::PointCloud<PointType>::Ptr final (new pcl::PointCloud<PointType>);
	pcl::transformPointCloud (*points1, *final, final_transform);
	//*final += *points2;
	pcl::io::savePCDFile(filename_1 + "_transformed.pcd", *final);
}

int main()
{
	int input_key, key;
	bool isExeced = false;

	func funcList[] = {
		pointcloud_demo, 1,
		pointcloud_process, 2,
		keypoints_demo, 3,
		normals_demo, 4,
		vfh_demo, 5,
		remove_nan, 6,
		coorespondences_demo, 7,
		icp_demo, 8,
		point_to_plane_icp, 9,
		NULL, '\0'
	};
	func *functionPtr;

	while (1) {
		prtUsage();
		cin >> input_key;

		isExeced = false;
		functionPtr = funcList;
		for (; key = functionPtr->key; ++functionPtr) {
			if (input_key == key) {
				functionPtr->fptr();
				isExeced = true;
				break;
			}
		}

		system("CLS");
		if (isExeced == false)
			cerr << "No such function!" << endl << endl;
	}

	return 0;
}