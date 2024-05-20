import numpy as np
import pdb

class grid2D():
    def __init__(self, minX, minY, maxX, maxY, dimensions):
        self.bounds=[[minX,minY],[maxX,maxY]]
        self.num_rows=int(dimensions[1])
        self.num_cols=int(dimensions[0])
        self.cell_width=(self.bounds[1][0]-self.bounds[0][0])/self.num_cols
        self.cell_height=(self.bounds[1][1]-self.bounds[0][1])/self.num_rows

    def is_xy_in_bounds(self, X, Y):
        if X<self.bounds[0,0] or X>self.bounds[1,0]:
            return False
        if Y<self.bounds[0,1] or X>self.bounds[1,1]:
            return False
        return True

    def is_rc_in_bounds(self, row, col):
        if row<0 or row>=self.num_rows:
            return False
        if col<0 or col>=self.num_cols:
            return False
        return True
    
    def get_row_col(self, X, Y):
        col=np.floor((X-self.bounds[0][0])/self.cell_width).astype(int)
        row=np.floor((Y-self.bounds[0][1])/self.cell_height).astype(int)
        return row, col

class average_grid_dict(grid2D):
    def __init__(self, minX, minY, maxX, maxY, dimensions):
        super().__init__(minX, minY, maxX, maxY, dimensions)
        self.create_empty_grid()

    def create_empty_grid(self):
        self.grid={}
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                self.grid[self.get_key(row,col)]=[]
    
    def get_key(self, row, col):
        return str(row)+"_"+str(col)
    
    def add_value(self, X, Y, value, cell_radius=None):
        RC=self.get_row_col(X,Y)
        if not self.is_rc_in_bounds(RC[0],RC[1]):
            print("Gaussian centroid outside region - skipping")
        if cell_radius is None:
            self.grid[self.get_key(RC[0],RC[1])].append(value)
        else:
            for row in range(RC[0]-cell_radius,RC[0]+cell_radius,1):
                for col in range(RC[1]-cell_radius,RC[1]+cell_radius,1):
                    if self.is_rc_in_bounds(row,col):
                        self.grid[self.get_key(row,col)].append(value)
    
    def get_grid_average(self):
        arr=np.zeros((self.num_rows, self.num_cols),dtype=float)
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                key=self.get_key(row,col)
                if len(self.grid[key])>0:
                    arr[row,col]=np.mean(self.grid[key])
        return arr

class average_grid(grid2D):
    def __init__(self, minX, minY, maxX, maxY, dimensions):
        super().__init__(minX, minY, maxX, maxY, dimensions)
        self.num_dim=0
        self.create_empty_grid()

    def get_index(self, row, col):
        return row*self.num_cols + col
        
    def create_empty_grid(self):
        self.grid=[]
        for idx in range(self.num_cols*self.num_rows):
            self.grid.append([])

    def add_value(self, X, Y, value, cell_radius=None):
        if type(value)==np.ndarray:
            if self.num_dim==0:
                self.num_dim=len(value)
            elif self.num_dim!=len(value):
                print("Cannot add data of different lengths to the grid")
                return
        elif self.num_dim==0:
            self.num_dim=1
                    
        RC=self.get_row_col(X,Y)
        if not self.is_rc_in_bounds(RC[0],RC[1]):
            print("Gaussian centroid outside region - skipping")
        if cell_radius is None:
            self.grid[self.get_index(RC[0],RC[1])].append(value)
        else:
            for row in range(RC[0]-cell_radius,RC[0]+cell_radius,1):
                for col in range(RC[1]-cell_radius,RC[1]+cell_radius,1):
                    if self.is_rc_in_bounds(row,col):
                        self.grid[self.get_index(row,col)].append(value)
    
    def get_grid_average(self):
        arr=np.zeros((self.num_rows, self.num_cols, self.num_dim),dtype=float)
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                key=self.get_index(row,col)
                if len(self.grid[key])>0:
                    arr[row,col]=np.mean(self.grid[key],0)
        return arr

class grid3D():
    def __init__(self, minXYZ, maxXYZ, shapeXYZ):
        self.minXYZ=np.array(minXYZ)
        self.maxXYZ=np.array(maxXYZ)
        self.shape=np.array(shapeXYZ)
        
        self.cell_delta=(self.maxXYZ-self.minXYZ)/self.shape

    def is_xyz_in_bounds(self, X, Y, Z):
        if X<self.minXYZ[0] or X>=self.maxXYZ[0]:
            return False
        if Y<self.minXYZ[1] or Y>=self.maxXYZ[1]:
            return False
        if Z<self.minXYZ[2] or Z>=self.maxXYZ[2]:
            return False
        return True

    def is_cell_in_bounds(self, cX, cY, cZ):
        if cX<0 or cX>=self.shape[0]:
            return False
        if cY<0 or cY>=self.shape[1]:
            return False
        if cZ<0 or cZ>=self.shape[2]:
            return False
        return True
    
    def get_ranges(self):
        xx=np.arange(self.minXYZ[0]+self.cell_delta[0]/2.0,self.maxXYZ[0],self.cell_delta[0])
        yy=np.arange(self.minXYZ[1]+self.cell_delta[1]/2.0,self.maxXYZ[1],self.cell_delta[1])
        zz=np.arange(self.minXYZ[2]+self.cell_delta[2]/2.0,self.maxXYZ[2],self.cell_delta[2])
        return xx,yy,zz
    
    def get_cell(self, X, Y, Z):
        cX=np.floor((X-self.minXYZ[0])/self.cell_delta[0]).astype(int)
        cY=np.floor((Y-self.minXYZ[1])/self.cell_delta[1]).astype(int)
        cZ=np.floor((Z-self.minXYZ[2])/self.cell_delta[2]).astype(int)
        return cX,cY,cZ
    
class evidence_grid3D(grid3D):
    def __init__(self, minXYZ, maxXYZ, gridDimensions, num_inference_dim):
        super().__init__(minXYZ, maxXYZ, gridDimensions)
        self.num_inference_dim=num_inference_dim
        self.grid=self.create_empty_grid()
    
    def load_raw_grid(self, raw_grid):
        if raw_grid.shape==self.grid.shape:
            self.grid=raw_grid
        else:
            print("Raw grid is the wrong shape")

    def create_empty_grid(self,type_=float):
        return np.zeros((self.num_inference_dim, self.shape[0], self.shape[1], self.shape[2]),dtype=type_)

    def add_evidence_cell(self, cell, vector):
        if self.is_cell_in_bounds(cell[0],cell[1],cell[2]) and vector.shape[0]==self.num_inference_dim:
            self.grid[:,cell[0],cell[1],cell[2]]+=vector

    def add_evidence_logodds(self, xyz, vector):
        cell=self.get_cell(xyz[0],xyz[1],xyz[2])
        vL = np.log(vector+1e-6)-np.log(1-vector+1e-6)
        self.add_evidence_cell(cell,vL)

    def add_ray(self, xyz, camera_pose, vector):
        cell_list=[]
        # xyz=np.matmul(camera_poseM, [0,0,depth,1])
        cell_xyz=self.get_cell(xyz[0],xyz[1],xyz[2])
        unit_vector=xyz-camera_pose
        depth=np.sqrt(np.power(xyz-camera_pose,2).sum())
        unit_vector/=depth
        
        for dpt in np.arange(0, depth, 0.02):
            P=camera_pose+unit_vector*dpt
            C=self.get_cell(P[0],P[1],P[2])

            if C not in cell_list and C!=cell_xyz:
                cell_list.append(C)
        
        v_reduced=np.ones((self.num_inference_dim),dtype=float)*-0.5
        for cell in cell_list:
            self.add_evidence_cell(cell, v_reduced)
        
        self.add_evidence_logodds(xyz, vector)            

    def get_thresholded_points(self, whichDim, threshold=0.0):
        pos=np.where(self.grid[whichDim,:,:,:]>threshold)
        xx,yy,zz=self.get_ranges()
        return np.vstack((xx[pos[0]],yy[pos[1]],zz[pos[2]],self.grid[whichDim,pos[0],pos[1],pos[2]]))

    # def add_probability_grid(self, xyz_matrix, p_grid):
    #     if xyz_matrix.shape[1]!=p_grid.shape[1] or xyz_matrix[2]!=p_grid.shape[2]:
    #         print("XYZ Matrix and P-grid should have same 1+2 dimensions")
    #         return False
    #     if xyz_matrix.shape[0]!=3:
    #         print("XYZ matrix should have size 3 in first dimension")
    #     if p_grid.shape[0]!=self.num_inference_dim:
    #         print("P_grid inference dimensions do not match this evidence grid")
        
    #     #Strategy - we are only want to update each cell only once with the maximum value
    #     #   from this image. So create a fresh evidence grid of the same size and add
    #     #   data using the max option
    #     p_comp=self.create_empty_grid(self.num_inference_dim)
    #     is_updated=self.create_empty_grid(1,np.uint8)



