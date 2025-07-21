import numpy as np
import math

gt = []
with open("ground_truth_raw.txt","r") as f:
    gt = f.readlines()

with open("groundtruth.txt","w") as f:

    f.write("# timestamp tx ty tz qx qy qz qw\n")
    
    for fi,frame in enumerate(gt):
        eyeX,eyeY,eyeZ,targetX,targetY,targetZ,uX,uY,uZ = [float(x) for x in frame.split()]

        # Compute quaternion

        #forward
        fX = (targetX - eyeX)
        fY = (targetY - eyeY)
        fZ = (targetZ - eyeZ)

        #right
        rightVec = -np.cross([fX,fY,fZ],[uX,uY,uZ])
        rX,rY,rZ =rightVec / np.linalg.norm(rightVec)

        #Should be normalized and orthonormal (1,1,1,0,0,0)
        # print(np.linalg.norm([fX,fY,fZ]),np.linalg.norm([rX,rY,rZ]),np.linalg.norm([uX,uY,uZ]), \
        #       np.dot([fX,fY,fZ],[uX,uY,uZ]),
        #       np.dot([rX,rY,rZ],[uX,uY,uZ]),
        #       np.dot([fX,fY,fZ],[rX,rY,rZ]))

        qx,qy,qz,qw = 0,0,0,0
        trace = rX + uY + fZ

        if trace > 0.0:
            s = 2.0 * math.sqrt(trace + 1.0)
            qw = 0.25 * s
            qx = (uZ - fY) / s
            qy = (fX - rZ) / s
            qz = (rY - uX) / s
        else:
            if (rX > uY) and (rX > fZ):
                s = 2.0 * math.sqrt(1.0 + rX - uY - fZ)
                qw = (uZ - fY) / s
                qx = 0.25 * s
                qy = (uX + rY) / s
                qz = (fX + rZ) / s
            elif uY > fZ:
                s = 2.0 * math.sqrt(1.0 + uY - rX - fZ)
                qw = (fX - rZ) / s
                qx = (uX + rY) / s
                qy = 0.25 * s
                qz = (fY + uZ) / s
            else:
                s = 2.0 * math.sqrt(1.0 + fZ - rX - uY)
                qw = (rY - uX) / s
                qx = (fX + rZ) / s
                qy = (fY + uZ) / s
                qz = 0.25 * s
        
        f.write(f"{fi+1.0:07.1f} {eyeX} {eyeY} {eyeZ} {qx} {qy} {qz} {qw}\n")

    print("Processed ground truth")