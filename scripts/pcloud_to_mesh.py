import open3d as o3d
import argparse
import numpy as np

def mesh_from_pcloud(pcd, 
                     pct_samples=0.2,
                     mesh_type='poisson',
                     poisson_octree_depth=9, 
                     poisson_low_support=0.01,
                     ball_radius=0.01):
    # Downsample the pointcloud - voxel based method is generating bad surface normals
    # downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    downpcd = pcd.random_down_sample(pct_samples)

    # Need the surface normals
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    # Create the color mesh
    if mesh_type=='poisson':
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(downpcd, depth=poisson_octree_depth)
        # Clean up faces with low support
        vertices_to_remove = densities < np.quantile(densities, poisson_low_support)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    elif mesh_type=='ball':
        radii = [ball_radius, 2*ball_radius, 4*ball_radius, 8*ball_radius]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(downpcd, o3d.utility.DoubleVector(radii))
    else:
        print("Unknown mesh type: " + mesh_type)
        import sys
        sys.exit(-1)


    # Visualize and save
    o3d.visualization.draw_geometries([mesh]) 
    return mesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_pcloud',type=str,help='location of raw point cloud')
    parser.add_argument('output_mesh',type=str,help='output mesh file')
    parser.add_argument('--pct-samples',type=float,default=0.2, help='how much to down sample the input file')
    parser.add_argument('--mesh-method',type=str,default='poisson',help='method to generate the resulting mesh [poisson, ball]')
    parser.add_argument('--radius-start',type=float,default=0.01,help="Starting radii for use with the pcloud (only with ball type mesh method)")
    parser.add_argument('--octree-depth',type=int,default=9,help="Octree depth when creating the mesh (only with poisson type mesh method)")
    parser.add_argument('--density-thresh',type=int,default=0.05,help="Minimum density to include from poisson based mesh method (only wih poisson)")
    args = parser.parse_args()

    pcd=o3d.io.read_point_cloud(args.input_pcloud)
    
    mesh=mesh_from_pcloud(pcd,args.pct_samples,args.mesh_method,args.octree_depth,args.density_thresh,args.radius_start)

    o3d.io.write_triangle_mesh(args.output_mesh,mesh)
