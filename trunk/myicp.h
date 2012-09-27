#include <pcl/pcl_base.h>
#include <pcl/point_types.h>

class MYICP
{
	public:
		typedef Eigen::Matrix<float, 4, 4> Matrix4;

		MYICP() {};

		Eigen::Matrix4f align();

		float point2plane_distance(
			const pcl::PointNormal point,
			const pcl::PointNormal plane) const;

		void estimateRigidTransformation(
			const pcl::PointCloud<pcl::PointNormal> &cloud_src,
			const std::vector<int> &indices_src,
            const pcl::PointCloud<pcl::PointNormal> &cloud_tgt,
			const std::vector<int> &indices_tgt,
            Matrix4 &transformation_matrix) const;
};