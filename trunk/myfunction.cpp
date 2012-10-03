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
	cout << "(7) coorespondences_demo" << endl;
	cout << "(8) icp_demo" << endl;
	cout << "(9) point_to_plane_icp" << endl << endl;
	cout << "Input exec function: ";
}


//
// 以 KSEARCH_SIZE 計算 points 的 normal, 結果放在 normals
//
void compute_surface_normals(pcl::PointCloud<PointType>::Ptr &points, pcl::PointCloud<pcl::Normal>::Ptr &normals)
{
	pcl::NormalEstimationOMP<PointType, pcl::Normal> n;
	n.setKSearch(KSEARCH_SIZE);
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
	pcl::visualization::PCLVisualizer viewer;
	viewer.addPointCloud(points, "points");

	viewer.addPointCloudNormals<PointType, pcl::Normal> (points, normals, 15, 0.006, "normals");

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
// 對 pointcloude 做特定處理, 測試用
//
void pointcloud_process()
{
	string filename, savename;
	cout << "Input process filename: ";
	cin >> filename;

	// stl model test
	///*
	pcl::PolygonMesh::Ptr model_mesh(new pcl::PolygonMesh);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointNormal>::Ptr plane_point (new pcl::PointCloud<pcl::PointNormal>);
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

	pcl::io::loadPolygonFileSTL(filename, *model_mesh);
	pcl::fromROSMsg(model_mesh->cloud, *cloud);

	cout << "number of model vertex: " << cloud->points.size() << endl;
	cout << "number of model surfaces: " << model_mesh->polygons.size() << endl;

	//pcl::visualization::PCLVisualizer viewer;
	//viewer.setBackgroundColor(0.2, 0.3, 0.4);

	//viewer.addPolygonMesh(*model_mesh, "polygon");

	//pcl::PointCloud<pcl::PointXYZ>::Ptr model (new pcl::PointCloud<pcl::PointXYZ>);
	//viewer.addPointCloud(model, "cloud");
	//viewer.spin();

	compute_surface_normals(cloud, normals);
	pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);


	// test texture mapping
	pcl::TextureMapping<pcl::PointXYZ> tm;

	tm.setF(0.01);
	tm.setVectorField(1, 0, 0);

	pcl::TexMaterial tex_material;
	tex_material.tex_Ka.r = 0.2f;
    tex_material.tex_Ka.g = 0.2f;
    tex_material.tex_Ka.b = 0.2f;

    tex_material.tex_Kd.r = 0.8f;
    tex_material.tex_Kd.g = 0.8f;
    tex_material.tex_Kd.b = 0.8f;

    tex_material.tex_Ks.r = 1.0f;
    tex_material.tex_Ks.g = 1.0f;
    tex_material.tex_Ks.b = 1.0f;
    tex_material.tex_d = 1.0f;
    tex_material.tex_Ns = 0.0f;
    tex_material.tex_illum = 2;

	tm.setTextureMaterials(tex_material);
	std::vector<std::string> tex_files;
	tex_files.push_back("RGB.jpg");

	mesh_convert_plane(model_mesh, plane_point);
	pcl::TextureMesh tex_mesh;
	tex_mesh.header = model_mesh->header;
	tex_mesh.cloud = model_mesh->cloud;
	tex_mesh.tex_polygons.push_back(model_mesh->polygons);


	tm.setTextureFiles(tex_files);
	//tm.mapTexture2Mesh(tex_mesh);
	tm.mapTexture2MeshUV(tex_mesh);

	pcl::toROSMsg(*cloud_with_normals, tex_mesh.cloud);
	//pcl::io::saveOBJFile("tex_mesh.obj", tex_mesh, 8);
	//cout << "tex_mesh.tex_coordinates.size: " << tex_mesh.tex_coordinates.size() << endl;
	for (int i = 0; i < tex_mesh.tex_coordinates.size(); ++i) {
		//cout << "tex_mesh.tex_coordinates[0].size: " << tex_mesh.tex_coordinates[0].size() << endl;
		//for (int j = 0; j < tex_mesh.tex_coordinates[0].size(); ++j) {
		//	cout << tex_mesh.tex_coordinates[0][j][0] << endl;
		//	cout << tex_mesh.tex_coordinates[0][j][1] << endl;
		//}
	}
	cout << "done" << endl;
	char c;
	cin >> c;
	//mesh_convert_plane(model_mesh, plane_point);
	//pcl::io::savePCDFileASCII("model_cloud.pcd", *cloud);
	//pcl::io::savePCDFileASCII("plane_point.pcd", *plane_point);

	//*/

	
	// old test
	/*
	// Create new point clouds to hold data
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr points (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr points_out (new pcl::PointCloud<pcl::PointXYZ>);

	// Load point cloud
	pcl::io::loadPCDFile(filename, *points);

	pcl::PointCloud<int> sampled_indices;
	pcl::copyPointCloud<pcl::PointXYZRGB, pcl::PointXYZ>(*points, *points_out);

	points_out->width = points->width;
	points_out->height = 1;
	points_out->resize(points_out->width);
	size_t cnt = 0;
	float min = points->points[0].z, max = points->points[0].z;
	for (size_t i = 0; i < points->size(); ++i) {
		if (points->points[i].z > max) max = points->points[i].z;
		if (points->points[i].z < min) min = points->points[i].z;
	}
	cout << min << " " << max << endl;
	float mid = (max + min) / 2;
	for (size_t i = 0; i < points->size(); ++i) {
		if (points->points[i].z < mid) {
			points_out->points[cnt] = points->points[i];
			++cnt;
		}
	}
	points_out->width = cnt;
	points_out->resize(points_out->width);
	*/
	//pcl::visualization::PCLVisualizer viewer;
	//viewer.addPointCloud(points_out, "points");
	//viewer.spin();

	//cout << "Input save filename: ";
	//cin >> savename;
	//pcl::io::savePCDFile(savename, *points_out);
}