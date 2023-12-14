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
        if type(value)==np.array:
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