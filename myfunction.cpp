#include "myfunction.h"

//
// �ϥλ���
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
// �H KSEARCH_SIZE �p�� points �� normal, ���G��b normals
//
void compute_surface_normals(pcl::PointCloud<PointType>::Ptr &points, pcl::PointCloud<pcl::Normal>::Ptr &normals)
{
	pcl::NormalEstimationOMP<PointType, pcl::Normal> n;
	n.setKSearch(KSEARCH_SIZE);
	n.setInputCloud(points);
	n.compute(*normals);
}


//
// �ھ� DOWNSAMPLE_RATE downsample points �� size, ���G��b points_downsample
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
// �� PCLVisualizer �� normals �����G
//
void visualize_normals(const pcl::PointCloud<PointType>::Ptr points, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
	// Add points to the visualizer
	pcl::visualization::PCLVisualizer viewer;
	viewer.addPointCloud(points, "points");

	viewer.addPointCloudNormals<PointType, pcl::Normal> (points, normals, 15, 0.006, "normals");

	// Give control over to the visualizer
	viewer.spin();
}


//
// �� PCLVisualizer �� pointcloud
//
void visualize_pointcloud(const pcl::PointCloud<PointType>::Ptr points)
{
	// Add clouds to vizualizer
	pcl::visualization::PCLVisualizer viewer;
	viewer.addPointCloud(points, "points");
	viewer.spin();
}


//
// normals �� demo �{��
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
// pointcloud �� demo �{��
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
// �p�� pointcloud �� VFH, �s�� vfh.pcd
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
// �� mesh model ���C�@�� face ��X�����I�P normal, return point at the center of face and its normal
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
// �� pointcloude ���S�w�B�z, ���ե�
//
void pointcloud_process()
{
	string filename, savename;
	cout << "Input process filename: ";
	cin >> filename;

	///*
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointNormal>::Ptr plane_point (new pcl::PointCloud<pcl::PointNormal>);
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

	pcl::io::loadPCDFile(filename, *cloud);

	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);

	double nbr_dist = 0;
	for (int i = 0; i < cloud->size(); ++i) {
		vector<int> pointNbrIdx;
		vector<float> pointNbrSquaredDistance;
		if (kdtree.nearestKSearch(i, 2, pointNbrIdx, pointNbrSquaredDistance) <= 0) {
			cout << "error happen" << endl;
		}
		cout << i << "  " << pointNbrIdx[0] << endl;
		nbr_dist += pointNbrSquaredDistance[0];
	}

	//cout << pointNbrSquaredDistance << endl;
	cout << nbr_dist / cloud->size() << endl;
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