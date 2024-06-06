import open3d as o3d

if __name__=="__main__":
    bowl_mesh=o3d.io.read_triangle_mesh("bowl.obj")
    # dustpan scale to 0.001
    # dustpan traslate to bowl center
    dustpan_mesh=o3d.io.read_triangle_mesh("dustpan.obj")
    # dustpan_mesh.scale(0.01,center=dustpan_mesh.get_center())
    # dustpan_mesh.translate(bowl_mesh.get_center()-dustpan_mesh.get_center())
    # move dustpan to the origin
    dustpan_mesh.translate(-dustpan_mesh.get_center())

    bowl_mesh.compute_vertex_normals()
    dustpan_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([dustpan_mesh,bowl_mesh])

    # output dustpan mesh
    o3d.io.write_triangle_mesh("dustpan.obj",dustpan_mesh)