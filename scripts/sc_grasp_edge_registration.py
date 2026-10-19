"""Bounded RGB edge registration of a known rigid SC housing mesh.

Initialization is a predicted grasp pose. Fixed original-image crop locations
and camera calibration are reused; no simulator geometry chooses an ROI.
"""
import cv2
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from render_sc_grasp_label_mesh_check import clouds
from evaluate_sc_native_triangulation import FOCAL


class SCGraspEdgeRegistration:
    def __init__(self, optics, origins, rotation_only=False):
        self.rotation_only=rotation_only
        self.cameras=('center','left','right')
        self.optics={c:np.linalg.inv(np.asarray(optics[c])) for c in self.cameras}
        self.origins=np.asarray(origins)
        pose=list(map(float,ET.parse('aic_assets/models/SC Plug/model.sdf').getroot().find("model/link[@name='sc_tip_link']/pose").text.split()))
        mesh_tip=np.eye(4);mesh_tip[:3,3]=pose[:3];mesh_tip[:3,:3]=Rotation.from_euler('xyz',pose[3:]).as_matrix()
        tip_mesh=np.linalg.inv(mesh_tip)
        vertices=np.vstack(clouds(Path('aic_assets/models/SC Plug/sc_plug_visual.glb')))
        self.vertices=np.unique(vertices@tip_mesh[:3,:3].T+tip_mesh[:3,3],axis=0)

    def project(self,points,pose,camera,index):
        transform=self.optics[camera]@pose
        xyz=points@transform[:3,:3].T+transform[:3,3]
        return xyz[:,:2]/xyz[:,2:]*FOCAL+[576.,512.]-self.origins[index]

    def refine(self,patches,initial):
        started=time.perf_counter();segments=[];fields=[];retained=[]
        for ci,c in enumerate(self.cameras):
            image=np.asarray(patches[ci]);gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            edge=cv2.Canny(gray,30,90)
            field=cv2.distanceTransform(255-edge,cv2.DIST_L2,cv2.DIST_MASK_PRECISE).astype(float)
            vertices=self.vertices
            xy=self.project(vertices,initial,c,ci);hull=ConvexHull(xy).vertices
            points=[]
            for i,j in zip(hull,np.roll(hull,-1)):
                count=max(2,int(np.linalg.norm(xy[i]-xy[j])/2))
                points.extend((1-f)*vertices[i]+f*vertices[j] for f in np.linspace(0,1,count,endpoint=False))
            points=np.array(points);uv=self.project(points,initial,c,ci)
            valid=(uv[:,0]>3)&(uv[:,0]<188)&(uv[:,1]>3)&(uv[:,1]<188)
            # Black gripper occlusion hides some predicted housing contours.
            # Use observed local intensity only to reject those measurements.
            bright=maximum_filter(gray,size=7)
            intensity=map_coordinates(bright.astype(float),[uv[:,1],uv[:,0]],order=1,mode='constant',cval=0)
            valid &= intensity>145
            segments.append(points[valid]);fields.append(field);retained.append(int(valid.sum()))
        if sum(retained)<30:return initial.copy(),dict(accepted=False,reason='insufficient_observed_housing_edges',retained=retained)
        def transform(delta):
            pose=initial.copy();pose[:3,3]+=delta[:3]/1000 if not self.rotation_only else 0
            pose[:3,:3]=Rotation.from_rotvec(np.deg2rad(delta[3:])).as_matrix()@initial[:3,:3]
            return pose
        def residual(delta):
            pose=transform(delta);out=[]
            for ci,c in enumerate(self.cameras):
                uv=self.project(segments[ci],pose,c,ci)
                out.extend(map_coordinates(fields[ci],[uv[:,1],uv[:,0]],order=1,mode='constant',cval=20).tolist())
            # Weak prior limits contour ambiguities; bounds provide the hard
            # limit. No ground-truth pose or error is consulted here.
            out.extend((delta[:3]/3).tolist());out.extend((delta[3:]/8).tolist())
            return np.asarray(out)
        before=residual(np.zeros(6))
        if self.rotation_only:
            result=least_squares(lambda q:residual(np.r_[np.zeros(3),q]),np.zeros(3),bounds=([-8]*3,[8]*3),loss='soft_l1',f_scale=1.,diff_step=.001,max_nfev=60)
            delta=np.r_[np.zeros(3),result.x]
        else:
            result=least_squares(residual,np.zeros(6),bounds=([-3]*3+[-8]*3,[3]*3+[8]*3),loss='soft_l1',f_scale=1.,diff_step=.001,max_nfev=60)
            delta=result.x
        final=transform(delta);after=residual(delta)
        return final,dict(accepted=True,retained=retained,initial_pixel_residual_p50_p95=np.percentile(before[:-6],[50,95]).tolist(),final_pixel_residual_p50_p95=np.percentile(after[:-6],[50,95]).tolist(),delta_mm_deg=delta.tolist(),evaluations=int(result.nfev),latency_ms=(time.perf_counter()-started)*1000,geometry_runtime_input=False)
