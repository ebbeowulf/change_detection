import numpy as np

class camera_params():
    def __init__(self, height, width, fx, fy, cx, cy, rot_matrix):
        self.height=int(height)
        self.width=int(width)
        self.fx=fx
        self.fy=fy
        self.cx=cx
        self.cy=cy
        self.rot_matrix=rot_matrix
        self.K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    
    def globalXYZ_to_imageRC(self, tX, tY, tZ, globalM:np.array):
        invR=np.linalg.inv(globalM)
        Vi=np.matmul(invR,[tX, tY, tZ, 1])
        col=Vi[0]*self.fx/Vi[2]+self.cx
        row=Vi[1]*self.fy/Vi[2]+self.cy
        return row,col
