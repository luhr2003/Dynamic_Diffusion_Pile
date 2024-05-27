import open3d as o3d

if __name__=="__main__":
    bean_mesh=o3d.io.read_triangle_mesh("/home/isaac/dyn-res-pile-manip/coffee_bean.ply")
    bean_mesh.compute_vertex_normals()
    dustpan_mesh=o3d.io.read_triangle_mesh("/home/isaac/dyn-res-pile-manip/dustpan2.ply")
    dustpan_mesh.compute_vertex_normals()
    dustpan_scale=0.005
    dustpan_mesh.scale(dustpan_scale,center=dustpan_mesh.get_center())
    dustpan_center=dustpan_mesh.get_center()
    dustpan_mesh.translate(-dustpan_center)

    o3d.visualization.draw_geometries([bean_mesh,dustpan_mesh])

    # o3d.io.write_triangle_mesh("/home/isaac/dyn-res-pile-manip/PyFleX/data/dustpan.ply",dustpan_mesh,write_ascii=True)
# 
    bean_pcd=o3d.geometry.PointCloud()
    bean_pcd.points=bean_mesh.vertices


    dustpan_pcd=o3d.geometry.PointCloud()
    dustpan_pcd.points=dustpan_mesh.vertices

    o3d.visualization.draw_geometries([bean_pcd,dustpan_pcd])