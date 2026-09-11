"""Reproduce the analytical examples in the q1 review revision; not a full solver.

Run from the repository root: python src/q1/verify_revision_examples.py
Requires numpy; outputs results/tables/q1_revision_validation.json.
"""
from pathlib import Path
import json
import math
from fractions import Fraction
import random
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

# Jung screening uses exact squared diameters, so rounded displays cannot move a threshold.
def jung_screen(diameter_squared):
    if diameter_squared <= 1200:
        return 'exists'
    if diameter_squared > 1600:
        return 'impossible_for_entire_region'
    return 'undetermined'

jung_cases = []
for label, shape, d2 in [
    ('equilateral_jung_boundary', 'equilateral', Fraction(1200)),
    ('square_d36', 'square', Fraction(1296)),
    ('equilateral_d36', 'equilateral', Fraction(1296)),
    ('square_d40', 'square', Fraction(1600)),
    ('equilateral_d40', 'equilateral', Fraction(1600)),
]:
    diameter = math.sqrt(float(d2))
    if shape == 'equilateral':
        points = np.array([[0, 0], [diameter, 0],
                           [diameter/2, diameter*math.sqrt(3)/2]])
        center = points.mean(axis=0)
        exact_radius_squared = d2 / 3
    else:
        side = diameter / math.sqrt(2)
        points = np.array([[0, 0], [side, 0], [side, side], [0, side]])
        center = np.array([side/2, side/2])
        exact_radius_squared = d2 / 4
    pair_squared = ((points[:, None, :] - points[None, :, :])**2).sum(axis=2)
    radius_squared = float(((points-center)**2).sum(axis=1).max())
    assert abs(float(pair_squared.max())-float(d2)) < 1e-9
    assert abs(radius_squared-float(exact_radius_squared)) < 1e-9
    assert d2/4 <= exact_radius_squared <= d2/3
    a, b = np.unravel_index(pair_squared.argmax(), pair_squared.shape)
    midpoint = (points[a]+points[b])/2
    midpoint_radius_squared = float(((points-midpoint)**2).sum(axis=1).max())
    assert (midpoint_radius_squared <= float(d2)/4 + 1e-9) == (shape == 'square')
    jung_cases.append(dict(
        case=label, diameter_m=diameter,
        radius_m=math.sqrt(float(exact_radius_squared)),
        jung_screen=jung_screen(d2),
        can_cover_with_20m=exact_radius_squared <= 400,
        diameter_circle_covers=shape == 'square',
        radius_squared_check_error=abs(radius_squared-float(exact_radius_squared)),
    ))

threshold_checks = []
for d2_text, expected in [
    ('1199.99999999', 'exists'), ('1200', 'exists'),
    ('1200.00000001', 'undetermined'), ('1599.99999999', 'undetermined'),
    ('1600', 'undetermined'), ('1600.00000001', 'impossible_for_entire_region'),
]:
    result = jung_screen(Fraction(d2_text))
    assert result == expected
    threshold_checks.append(dict(diameter_squared_m2=d2_text, screen=result))
checks['jung_examples'] = jung_cases
checks['jung_squared_thresholds'] = threshold_checks

# Oriented arc fixtures: topology is input, never inferred as the shorter arc.
# Compare the analytic Green term with an independently closed chord polygon.
arc_checks = []
for label, start_deg, end_deg, full, expected_sweep in [
    ('atan2_seam', 179, -179, False, 2),
    ('zero_seam', 359, 1, False, 2),
    ('major_arc', 1, 359, False, 358),
    ('zero_arc', 25, 25, False, 0),
    ('full_circle', 25, 25, True, 360),
    ('semicircle', 270, 90, False, 180),
    ('over_semicircle', 270, 91, False, 181),
    ('under_semicircle', 270, 89, False, 179),
]:
    start_deg %= 360
    sweep_deg = 360 if full else (end_deg-start_deg) % 360
    assert sweep_deg == expected_sweep
    a, sweep = math.radians(start_deg), math.radians(sweep_deg)
    b, radius = a+sweep, 5
    cx, cy = 3, -2
    term = (radius*cx*(math.sin(b)-math.sin(a))
            + radius*cy*(math.cos(a)-math.cos(b)) + radius**2*sweep)/2
    angles = np.linspace(a, b, 20001)
    points = np.column_stack((cx+radius*np.cos(angles), cy+radius*np.sin(angles)))
    # Add the returning chord to both representations, closing the boundary.
    chord = float(points[-1, 0]*points[0, 1]-points[-1, 1]*points[0, 0])/2
    following = np.roll(points, -1, axis=0)
    polygon_area = float(np.sum(points[:, 0]*following[:, 1]
                                - following[:, 0]*points[:, 1])/2)
    exact_area = term+chord
    assert exact_area >= -1e-12
    assert abs(exact_area-polygon_area) < 2e-6
    # Split at 360 degrees to verify that the branch cut changes no integral.
    def area_term(lo, hi):
        return (radius*cx*(math.sin(hi)-math.sin(lo))
                + radius*cy*(math.cos(lo)-math.cos(hi)) + radius**2*(hi-lo))/2
    split = 2*math.pi
    split_term = area_term(a, split)+area_term(split, b) if a < split < b else term
    assert abs(term-split_term) < 1e-12
    antipodal = sweep_deg >= 180
    if antipodal:
        pair = np.array([[cx+radius*math.cos(a), cy+radius*math.sin(a)],
                         [cx+radius*math.cos(a+math.pi), cy+radius*math.sin(a+math.pi)]])
        assert abs(float(np.linalg.norm(pair[0]-pair[1]))-2*radius) < 1e-12
    arc_checks.append(dict(case=label, sweep_deg=sweep_deg,
                           has_antipodal_pair=antipodal,
                           green_vs_polygon_error_m2=abs(exact_area-polygon_area)))
checks['oriented_arc_examples'] = arc_checks

# Two separately retained arcs can have an antipodal pair even though each is short.
assert 10 <= 15 <= 20 and 190 <= 195 <= 200 and (195-15) % 360 == 180
opposite_points = 1800*np.column_stack((np.cos(np.radians([15, 195])),
                                       np.sin(np.radians([15, 195]))))
assert abs(float(np.linalg.norm(opposite_points[0]-opposite_points[1]))-3600) < 1e-9
checks['separate_arc_antipodes'] = dict(arcs_deg=[[10, 20], [190, 200]],
                                       witness_deg=[15, 195], diameter_m=3600)

# Exact rational segment endpoints demonstrate why the rounded output needs a check.
left, right = Fraction('-19.996'), Fraction('20.004')
optimal_center, optimal_radius = (left+right)/2, (right-left)/2
output_center = Fraction(0)  # 0.004 rounded to two decimal places.
output_radius = max(abs(left-output_center), abs(right-output_center))
assert optimal_radius == 20 and output_radius > 20
checks['rounded_center_counterexample'] = dict(
    optimal_center_m=float(optimal_center), optimal_radius_m=float(optimal_radius),
    output_center_m=float(output_center), output_radius_m=float(output_radius),
    conclusion='exists_but_output_not_clear_ready')

# Predictive geometry at the center of Omega: the reception disk is not redundant.
theta_half = math.radians(1)
prior_area = math.pi*1800**2
predictive = {'direction': 1500**2*theta_half, 'near': math.pi*5**2,
              'no_signal': prior_area}
arc_angle = np.linspace(-theta_half, theta_half, 20001)
boundary = np.vstack(([0, 0], np.column_stack((1500*np.cos(arc_angle),
                                               1500*np.sin(arc_angle)))))
next_boundary = np.roll(boundary, -1, axis=0)
direction_area_check = float(np.sum(boundary[:, 0]*next_boundary[:, 1]
                                   - next_boundary[:, 0]*boundary[:, 1])/2)
assert abs(direction_area_check-predictive['direction']) < 1e-6
assert predictive['direction'] < 1800**2*theta_half
assert all(0 <= value <= prior_area for value in predictive.values())
checks['predictive_area_examples'] = dict(
    prior_area_m2=prior_area, predicted_area_m2=predictive,
    reduction_m2={key: prior_area-value for key, value in predictive.items()},
    direction_polygon_error_m2=abs(direction_area_check-predictive['direction']),
    scope='convex_envelope_only_no_probability_model')

# A disk intersection remains convex; removing the near disk need not do so.
outer_radius, excluded_radius = 10, 5
witnesses = np.array([[-6, 0], [6, 0]])
assert np.all(np.linalg.norm(witnesses, axis=1) <= outer_radius)
assert np.all(np.linalg.norm(witnesses, axis=1) > excluded_radius)
assert np.linalg.norm(witnesses.mean(axis=0)) <= excluded_radius
checks['nonconvex_exclusion_example'] = dict(
    envelope_radius_m=outer_radius, excluded_radius_m=excluded_radius,
    feasible_points=witnesses.tolist(), midpoint_excluded=True)

# Section 2.2 caliper formulas, checked against exhaustive pairs using exact integers.
# This validates the diameter step, not floating predicates or region construction.
def turn(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def integer_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points
    halves = []
    for sequence in (points, reversed(points)):
        half = []
        for point in sequence:
            while len(half) >= 2 and turn(half[-2], half[-1], point) <= 0:
                half.pop()
            half.append(point)
        halves.append(half[:-1])
    return halves[0]+halves[1]


def calipers_integer(vertices):
    m = len(vertices)
    assert m >= 3
    assert all(turn(vertices[i-1], vertices[i], vertices[(i+1) % m]) > 0
               for i in range(m))
    def vertex(k):
        return vertices[(k-1) % m]
    def area(i, j):
        return turn(vertex(i), vertex(i+1), vertex(j))
    j, maximum, witness, advances, ties = 2, 0, None, 0, 0
    for i in range(1, m+1):
        while j+1 < i+m and area(i, j+1) > area(i, j):
            j += 1
            advances += 1
        candidates = [(i, j), (i+1, j)]
        if j+1 < i+m and area(i, j+1) == area(i, j):
            candidates += [(i, j+1), (i+1, j+1)]
            ties += 1
        for p, q in candidates:
            a, b = vertex(p), vertex(q)
            squared = (a[0]-b[0])**2+(a[1]-b[1])**2
            if squared > maximum:
                maximum, witness = squared, (a, b)
    assert advances < 2*m
    return maximum, witness, advances, ties


grid = [(x, y) for x in range(3) for y in range(3)]
point_sets = [[point for bit, point in enumerate(grid) if mask & (1 << bit)]
              for mask in range(1 << len(grid))]
rng = random.Random(20260911)
point_sets += [[(rng.randrange(-2000, 2001), rng.randrange(-2000, 2001))
               for _ in range(rng.randrange(3, 61))] for _ in range(200)]
point_sets += [[(0, 0), (4, 0), (4, 3), (0, 3)],
               [(0, 0), (10**12, 1), (10**12, 2), (0, 1)]]
polygons, runs, tie_events, max_advances = 0, 0, 0, 0
for points in point_sets:
    hull = integer_hull(points)
    if len(hull) < 3:
        continue
    polygons += 1
    reference = max((a[0]-b[0])**2+(a[1]-b[1])**2
                    for i, a in enumerate(hull) for b in hull[i+1:])
    # Every cyclic starting vertex tests the wraparound and monotone pointer.
    for shift in range(len(hull)):
        rotated = hull[shift:]+hull[:shift]
        value, witness, advances, ties = calipers_integer(rotated)
        assert value == reference
        a, b = witness
        assert (a[0]-b[0])**2+(a[1]-b[1])**2 == reference
        runs += 1
        tie_events += ties
        max_advances = max(max_advances, advances)
assert tie_events > 0
assert calipers_integer([(0, 0), (4, 0), (4, 3), (0, 3)])[0] == 25
checks['calipers_vs_exhaustive'] = dict(
    random_seed=20260911, grid_subsets=512, random_point_sets=200,
    special_point_sets=2, nondegenerate_polygons=polygons,
    cyclic_start_runs=runs, equal_area_events=tie_events,
    max_pointer_advances=max_advances, exact_squared_distance_discrepancy=0,
    scope='integer_polygon_diameter_only_not_full_localization_solver')

# A neighboring output grid point can work even when coordinate-wise rounding fails.
# All squared distances are exact rationals; the region is the endpoint segment.
grid_endpoints = [(Fraction('14.1449'), Fraction('14.1449')),
                  (Fraction('-14.1351'), Fraction('-14.1351'))]
def segment_radius_squared(center):
    return max(sum((p[k]-center[k])**2 for k in range(2)) for p in grid_endpoints)
rounded_center = (Fraction(0), Fraction(0))
neighbor = (Fraction('0.01'), Fraction(0))
assert segment_radius_squared(rounded_center) > 400
assert segment_radius_squared(neighbor) < 400
checks['output_grid_neighbor'] = dict(
    endpoints_m=[[float(x) for x in p] for p in grid_endpoints],
    grid_step_m=.01, continuous_center_m=[.0049, .0049],
    rounded_center_m=[0, 0], rounded_radius_m=math.sqrt(float(segment_radius_squared(rounded_center))),
    feasible_neighbor_m=[.01, 0], neighbor_radius_m=math.sqrt(float(segment_radius_squared(neighbor))),
    scope='certified_feasible_candidate_not_claimed_grid_optimum')
# For the earlier length-40 segment, any radius-20 circle has its unique midpoint.
assert optimal_radius == 20 and (optimal_center/Fraction('.01')).denominator != 1
checks['grid_boundary_impossible'] = dict(
    unique_center_m=float(optimal_center), grid_step_m=.01,
    proof='length_40_segment_requires_unique_midpoint_which_is_not_on_grid')
# The source-excluded center of the annulus still covers the whole outer envelope.
assert excluded_radius < outer_radius <= 20
checks['excluded_center_is_valid_action'] = dict(
    source_region='5 < norm(G) <= 10', action_center_m=[0, 0],
    certified_cover_radius_m=10, center_excluded_as_source=True,
    action_valid=True)

out = root / 'results/tables/q1_revision_validation.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(checks, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(checks, ensure_ascii=True, indent=2))
