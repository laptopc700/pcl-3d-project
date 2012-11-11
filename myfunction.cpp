#include "myfunction.h"

//
// 使用說明
//
void prtUsage()
{
	cout << "Select functions..." << endl << endl;
	cout << "(1) pointcloud_demo" << endl;
	cout << "(2) pointcloud_process" << endl;
	cout << "(3) keypoints_demo" << endl;
	cout << "(4) normals_demo" << endl;
	cout << "(5) vfh_demo" << endl;
	cout << "(6) remove_nan" << endl;
	cout << "(7) model_to_plane_point" << endl;
	cout << "(8) coorespondences_demo" << endl;
	cout << "(9) icp_demo" << endl;
	cout << "(10) point_to_plane_icp" << endl << endl;
	cout << "Input exec function: ";
}


//
// 以 KSEARCH_SIZE 計算 points 的 normal, 結果放在 normals
//
void compute_surface_normals(pcl::PointCloud<PointType>::Ptr &points, pcl::PointCloud<pcl::Normal>::Ptr &normals)
{
	pcl::NormalEstimationOMP<PointType, pcl::Normal> n;
	n.setRadiusSearch(0.005);
	//n.setKSearch(KSEARCH_SIZE);
	n.setInputCloud(points);
	n.compute(*normals);
}


//
// 根據 DOWNSAMPLE_RATE downsample points 的 size, 結果放在 points_downsample
//
void downsample(const pcl::PointCloud<PointType>::Ptr &points, pcl::PointCloud<PointType>::Ptr &points_downsample)
{
	pcl::PointCloud<int> sampled_indices;

	pcl::UniformSampling<PointType> uniform_sampling;
	uniform_sampling.setInputCloud(points);
	uniform_sampling.setRadiusSearch(DOWNSAMPLE_RATE);
	uniform_sampling.compute(sampled_indices);

	pcl::copyPointCloud (*points, sampled_indices.points, *points_downsample);
}


//
// 用 PCLVisualizer 看 normals 的分佈
//
void visualize_normals(const pcl::PointCloud<PointType>::Ptr points, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
	// Add points to the visualizer
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::copyPointCloud(*points, *points_rgb);
	for (size_t i = 0; i < points_rgb->size(); ++i) {
		points_rgb->points[i].r = 255;
		points_rgb->points[i].g = 0;
		points_rgb->points[i].b = 0;
	}
	pcl::visualization::PCLVisualizer viewer;
	viewer.addPointCloud(points_rgb, "points");
	viewer.addCoordinateSystem();

	viewer.addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (points_rgb, normals, 15, 0.006, "normals");

	// Give control over to the visualizer
	viewer.spin();
}


//
// 用 PCLVisualizer 看 pointcloud
//
void visualize_pointcloud(const pcl::PointCloud<PointType>::Ptr points)
{
	// Add clouds to vizualizer
	pcl::visualization::PCLVisualizer viewer;
	viewer.addPointCloud(points, "points");
	viewer.spin();
}


//
// normals 的 demo 程式
//
void normals_demo()
{
	string filename;
	cout << "Input filename: ";
	cin >> filename;

	// Create new point clouds to hold data
	pcl::PointCloud<PointType>::Ptr points (new pcl::PointCloud<PointType>);
	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);

	// Load point cloud
	pcl::io::loadPCDFile(filename, *points);

	// Compute surface normals
	compute_surface_normals(points, normals);

	// Visualize the points and normals
	visualize_normals(points, normals);
}


//
// pointcloud 的 demo 程式
//
void pointcloud_demo()
{
	string filename;
	cout << "Input filename: ";
	cin >> filename;

	// Create new point clouds to hold data
	pcl::PointCloud<PointType>::Ptr points (new pcl::PointCloud<PointType>);

	// Load point cloud
	pcl::io::loadPCDFile(filename, *points);

	// Visualize point cloud
	visualize_pointcloud(points);
}


//
// 計算 pointcloud 的 VFH, 存成 vfh.pcd
//
void vfh_demo()
{
	string filename;
	cout << "Input filename: ";
	cin >> filename;

	// Create new point clouds to hold data
	pcl::PointCloud<PointType>::Ptr points (new pcl::PointCloud<PointType>);
	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs(new pcl::PointCloud<pcl::VFHSignature308>);

	// Load point data
	pcl::io::loadPCDFile(filename, *points);

	// Compute normals
	compute_surface_normals(points, normals);

	// Create VFHEstimation
	pcl::VFHEstimation<PointType, pcl::Normal, pcl::VFHSignature308> vfh;
	vfh.setInputCloud(points);
	vfh.setInputNormals(normals);

	pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType>());
	vfh.setSearchMethod(tree);
	vfh.compute(*vfhs);

	// Save VFH
	pcl::io::savePCDFileASCII("vfh.pcd", *vfhs);
}


//
// 對 mesh model 的每一個 face 找出中心點與 normal, return point at the center of face and its normal
//
void mesh_convert_plane(const pcl::PolygonMesh::Ptr &mesh, const pcl::PointCloud<pcl::PointNormal>::Ptr &plane_point)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_cloud(new pcl::PointCloud<pcl::PointXYZ>); 
	pcl::fromROSMsg(mesh->cloud, *mesh_cloud);

	plane_point->width = mesh->polygons.size();
	plane_point->height = 1;
	plane_point->resize(plane_point->width);
	for (size_t i = 0; i < plane_point->width; ++i) {
		pcl::PointXYZ centroid;
		pcl::PointXYZ p1 = mesh_cloud->points[mesh->polygons[i].vertices[0]];
		pcl::PointXYZ p2 = mesh_cloud->points[mesh->polygons[i].vertices[1]];
		pcl::PointXYZ p3 = mesh_cloud->points[mesh->polygons[i].vertices[2]];

		Eigen::Vector3f v1(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z),
						v2(p3.x - p1.x, p3.y - p1.y, p3.z - p1.z);
		Eigen::Vector3f normal = v1.cross(v2).normalized();

		plane_point->points[i].x = (p1.x + p2.x + p3.x) / 3;
		plane_point->points[i].y = (p1.y + p2.y + p3.y) / 3;
		plane_point->points[i].z = (p1.z + p2.z + p3.z) / 3;
		plane_point->points[i].normal_x = normal[0];
		plane_point->points[i].normal_y = normal[1];
		plane_point->points[i].normal_z = normal[2];
		plane_point->points[i].curvature = 0;
	}
}


//
// 算出 model 所有 face 的中點，存成 plane_point.pcd ，vertex 存成 model_cloud.pcd
//
void model_to_plane_point()
{
	string filename;
	cout << "Input model filename: ";
	cin >> filename;

	pcl::PolygonMesh::Ptr model_mesh(new pcl::PolygonMesh);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointNormal>::Ptr plane_point (new pcl::PointCloud<pcl::PointNormal>);

	pcl::io::loadPolygonFileSTL(filename, *model_mesh);
	pcl::fromROSMsg(model_mesh->cloud, *cloud);

	//cout << "number of model vertex: " << cloud->points.size() << endl;
	//cout << "number of model surfaces: " << model_mesh->polygons.size() << endl;

	mesh_convert_plane(model_mesh, plane_point);

	pcl::io::savePCDFileASCII("model_cloud.pcd", *cloud);
	pcl::io::savePCDFileASCII("plane_point.pcd", *plane_point);
}

//
//
//
void filter(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
	pcl::RadiusOutlierRemoval<pcl::PointXYZ> radius_filter;
	radius_filter.setInputCloud(cloud);
	radius_filter.setRadiusSearch(0.01);
	radius_filter.setMinNeighborsInRadius(500);
	radius_filter.filter(*cloud);
}


//
// 對 pointcloude 做特定處理, 測試用
//
void pointcloud_process()
{
	string filename, savename;
	cout << "Input process filename: ";
	cin >> filename;
	//filename = "data_smooth.pcd";
	
	///*
	pcl::PointCloud<pcl::PointXYZ>::Ptr points (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr points_out (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr features (new pcl::PointCloud<pcl::FPFHSignature33>);

	pcl::io::loadPCDFile(filename, *points);

	compute_surface_normals(points, normals);
	pcl::io::savePCDFileASCII(filename + "_normals.pcd", *normals);

	// test
	/*
	pcl::copyPointCloud(*points, *points_rgb);
	for (int i = 0; i < points->size(); ++i) {
		points_rgb->points[i].r = 255;
		points_rgb->points[i].g = 255;
		points_rgb->points[i].b = 255;
	}
	vector<int> pointIdx;
	vector<float> pointSquDisIdx;

	pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
	kdtree.setInputCloud(points_rgb);
	if (kdtree.radiusSearch(12656, 0.005, pointIdx, pointSquDisIdx) <= 0) {
		exit(-1);
	}
	cout << pointIdx.size() << endl;
	for (int i = 0; i < pointIdx.size(); ++i) {
		points_rgb->points[pointIdx[i]].g = 0;
		points_rgb->points[pointIdx[i]].b = 0;
	}
	pcl::io::savePCDFileASCII("output.pcd", *points_rgb);
	*/
	// end test

	///*
	cout << "before filter: " << points->size() << endl;
	//filter(points);
	//cout << "after filter: " << points->size() << endl;

	pcl::copyPointCloud(*points, *points_rgb);
	compute_surface_normals(points, normals);

	vector<int> pointIdx;
	vector<float> pointSquDisIdx;
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(points);

	FILE *pFile;
	pFile = fopen("RGB.txt", "r");
	int r, g, b;
	vector<int> R;
	vector<int> G;
	vector<int> B;
	while (fscanf(pFile, "%d %d %d", &r, &g, &b) != EOF) {
		R.push_back(r);
		G.push_back(g);
		B.push_back(b);
	}
	fclose(pFile);

	//pcl::io::savePCDFileASCII("features.pcd", *features);

	Eigen::Vector4f pi, pj, ps, pt;
	Eigen::Vector4f ni, nj, ns, nt;
	pi[3] = pj[3] = ni[3] = nj[3] = ps[3] = pt[3] = ns[3] = nt[3] = 0.0f;
	vector<double> stDev;
	double stdev_max = 0.0f;
	for (size_t i = 0; i < points->size(); ++i) {
		double sum = 0.0f, mid_square_sum = 0.0f;
		float freq = 0.0f;
		float f11, f22, f33, f44;
		double f3_sum = 0.0f;
		double var, stdev, mean, mid_square, f3_mean;
		int cnt = 0;

		if (kdtree.radiusSearch(i, 0.005, pointIdx, pointSquDisIdx) <= 0) {
		//if (kdtree.nearestKSearch(i, 50, pointIdx, pointSquDisIdx) <= 0) {
			exit(-1);
		}

		for (size_t j = 1; j < pointIdx.size(); ++j) {
			pi[0] = points->points[pointIdx[j]].x;
			pi[1] = points->points[pointIdx[j]].y;
			pi[2] = points->points[pointIdx[j]].z;
			ni[0] = normals->points[pointIdx[j]].normal_x;
			ni[1] = normals->points[pointIdx[j]].normal_y;
			ni[2] = normals->points[pointIdx[j]].normal_z;

			for (size_t k = j + 1; k < pointIdx.size(); ++k) {
				pj[0] = points->points[pointIdx[k]].x;
				pj[1] = points->points[pointIdx[k]].y;
				pj[2] = points->points[pointIdx[k]].z;
				nj[0] = normals->points[pointIdx[k]].normal_x;
				nj[1] = normals->points[pointIdx[k]].normal_y;
				nj[2] = normals->points[pointIdx[k]].normal_z;

				pcl::computePairFeatures(pi, ni, pj, nj, f11, f22, f33, f44);
				f3_sum += f33;
				++cnt;
			}
			//freq = freq + features->points[i].histogram[j];
			//sum = sum + (features->points[i].histogram[j] * (j + 1));
			//mid_square_sum = mid_square_sum + (features->points[i].histogram[j] * (j + 1) * (j + 1));
		}

		f3_mean = (f3_sum + cnt ) / cnt / 2;
		points_rgb->points[i].r = R[f3_mean * 63];
		points_rgb->points[i].g = G[f3_mean * 63];
		points_rgb->points[i].b = B[f3_mean * 63];

	}
	pcl::io::savePCDFileASCII("output.pcd", *points_rgb);
	//*/


	/* shift point cloud */
	/*
	pcl::PointCloud<pcl::PointXYZ>::Ptr points1 (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr points2 (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile("foreground_10_transformed.pcd", *points1);
	pcl::io::loadPCDFile("model.pcd", *points2);
	double x = points2->points[144385].x - points1->points[9101].x;
	double y = points2->points[144385].y - points1->points[9101].y;
	double z = points2->points[144385].z - points1->points[9101].z;
	for (int i = 0; i < points1->size(); ++i) {
		points1->points[i].x += x;
		points1->points[i].y += y;
		points1->points[i].z += z;
	}
	pcl::io::savePCDFileASCII("shift.pcd", *points1);
	*/

	/*

	//pcl::visualization::PCLVisualizer viewer;
	//viewer.setBackgroundColor(0.2, 0.3, 0.4);

	//viewer.addPolygonMesh(*model_mesh, "polygon");

	//pcl::PointCloud<pcl::PointXYZ>::Ptr model (new pcl::PointCloud<pcl::PointXYZ>);
	//viewer.addPointCloud(model, "cloud");
	//viewer.spin();

	//*/

	//pcl::visualization::PCLVisualizer viewer;
	//viewer.addPointCloud(points_out, "points");
	//viewer.spin();

	//cout << "Input save filename: ";
	//cin >> savename;
	//pcl::io::savePCDFile(savename, *points_out);
}