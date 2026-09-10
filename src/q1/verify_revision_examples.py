"""Reproduce the analytical examples in the q1 review revision; not a full solver.

Run from the repository root: python src/q1/verify_revision_examples.py
Requires numpy; outputs results/tables/q1_revision_validation.json.
"""
from pathlib import Path
import json
import math
import numpy as np

root = Path(__file__).resolve().parents[2]
checks = {}

# Independent midpoint integration of Green boundary moments: full/right-half disks.
def arc_integrals(radius, start, end, n=200000):
    dt = (end-start)/n
    t = start + (np.arange(n)+0.5)*dt
    x, y = radius*np.cos(t), radius*np.sin(t)
    dx, dy = -radius*np.sin(t), radius*np.cos(t)
    area = float(np.sum(x*dy-y*dx)*dt/2)
    mx = float(np.sum(x*x*dy)*dt/2)
    my = float(-np.sum(y*y*dx)*dt/2)
    return area, mx/area, my/area

for radius, start, end, label in [
    (1800, 0, 2*math.pi, 'full_disk'),
    (1800, -math.pi/2, math.pi/2, 'half_disk'),
    (5, -math.pi/2, math.pi/2, 'near_half_disk')]:
    area, cx, cy = arc_integrals(radius, start, end)
    exact_area = (end-start)*radius*radius/2
    exact_cx = 0 if label == 'full_disk' else 4*radius/(3*math.pi)
    assert abs(area-exact_area) <= 1e-8
    assert abs(cx-exact_cx) <= 1e-8 and abs(cy) <= 1e-8
    checks[label] = dict(area_error_m2=abs(area-exact_area), centroid_error_m=math.hypot(cx-exact_cx, cy))

# Sector polygon converges from inside; shoelace centroid vs polar integral formula.
R, d = 1800, math.radians(1)
t = np.linspace(-d,d,10001)
v = np.vstack(([0,0], np.column_stack((R*np.cos(t),R*np.sin(t)))))
w = np.roll(v,-1,axis=0)
cross = v[:,0]*w[:,1]-w[:,0]*v[:,1]
area = cross.sum()/2
cg = ((v+w)*cross[:,None]).sum(axis=0)/(6*area)
exact_area = R*R*d
exact_cg = np.array([2*R*math.sin(d)/(3*d),0])
assert 0 <= exact_area-area <= 1e-5
assert np.linalg.norm(cg-exact_cg) <= 1e-7
center = np.array([R/(2*math.cos(d)),0])
cover_radius = float(np.linalg.norm(v-center,axis=1).max())
assert abs(cover_radius-center[0]) < 1e-9
assert cover_radius > R/2
checks['sector'] = dict(inscribed_area_gap_m2=float(exact_area-area), centroid_error_m=float(np.linalg.norm(cg-exact_cg)), radius_m=cover_radius)

# Independent line-intersection enumeration and exhaustive diameter for symmetric cases.
for deg in (.8,1,1.005,1.2):
    constraints = []
    for sx, sy, theta in ((0,0,45),(1000,0,135)):
        for angle, side in ((theta-deg,-1),(theta+deg,1)):
            r = math.radians(angle)
            normal = side*np.array([-math.sin(r),math.cos(r)])
            constraints.append((normal,float(normal@np.array([sx,sy]))))
    vertices=[]
    for i,(a,b) in enumerate(constraints):
        for c,e in constraints[i+1:]:
            mat=np.array([a,c])
            if abs(np.linalg.det(mat))<1e-12: continue
            point=np.linalg.solve(mat,np.array([b,e]))
            if all(normal@point <= rhs+1e-8 for normal,rhs in constraints):
                vertices.append(point)
    assert len(vertices)==4
    vertices=np.array(vertices)
    pair=vertices[:,None,:]-vertices[None,:,:]
    distances=np.linalg.norm(pair,axis=2)
    a,b=np.unravel_index(distances.argmax(),distances.shape)
    diameter=float(distances[a,b])
    mid=(vertices[a]+vertices[b])/2
    max_radius=float(np.linalg.norm(vertices-mid,axis=1).max())
    analytic=500*(math.tan(math.radians(45+deg))-math.tan(math.radians(45-deg)))
    assert abs(diameter-analytic)<1e-9
    assert abs(max_radius-diameter/2)<1e-9
    assert np.linalg.norm(vertices,axis=1).max()<1800
    checks[f'symmetric_{deg}'] = dict(diameter_m=diameter, difference_m=abs(diameter-analytic))

# Triangle classification (diameter midpoint): acute, right, obtuse.
for vertex,expect,label in [([.5,math.sqrt(3)/2],False,'acute'),([.5,.5],True,'right'),([.5,.25],True,'obtuse')]:
    q=np.array([[0,0],[1,0],vertex])
    sq=((q[:,None,:]-q[None,:,:])**2).sum(axis=2)
    a,b=np.unravel_index(sq.argmax(),sq.shape)
    mid=(q[a]+q[b])/2
    covered=bool(((q-mid)**2).sum(axis=1).max() <= sq[a,b]/4+1e-14)
    assert covered == expect
    checks[f'triangle_{label}']=dict(diameter_circle_covers=covered)

out = root / 'results/tables/q1_revision_validation.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(checks, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(checks, ensure_ascii=True, indent=2))
