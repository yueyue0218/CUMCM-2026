"""Bounded relocation search for an open, fixed-start sensing route."""
import math
from src.q4.q3_adapter import optimized_route, open_clear_route


def relocated_route(start, targets):
    if len(targets)<=10:
        return open_clear_route(start,targets)
    keys=sorted(targets)
    points=[start]+[targets[c] for c in keys]
    distances=[[math.dist(a,b) for b in points] for a in points]
    lookup={c:i+1 for i,c in enumerate(keys)}
    path=[0]+[lookup[c] for c in optimized_route(start,targets)]
    n=len(keys)
    for _ in range(25):
        best=(0.,None)
        for size in (1,2,3):
            for i in range(1,n-size+2):
                first,last=path[i],path[i+size-1]
                before=path[i-1]
                after=path[i+size] if i+size<=n else None
                removal=-distances[before][first]
                if after is not None:
                    removal+=distances[before][after]-distances[last][after]
                rest=path[:i]+path[i+size:]
                for j in range(1,len(rest)+1):
                    if j==i: continue
                    a=rest[j-1]
                    b=rest[j] if j<len(rest) else None
                    gain=removal+distances[a][first]
                    if b is not None:
                        gain+=distances[last][b]-distances[a][b]
                    if gain < best[0]-1e-8:
                        best=(gain,rest[:j]+path[i:i+size]+rest[j:])
        if best[1] is None:
            break
        path=best[1]
        for i in range(1,n):
            for j in range(i+1,n+1):
                old=distances[path[i-1]][path[i]]
                new=distances[path[i-1]][path[j]]
                if j<n:
                    old+=distances[path[j]][path[j+1]]
                    new+=distances[path[i]][path[j+1]]
                if new<old-1e-8:
                    path[i:j+1]=reversed(path[i:j+1])
    return [keys[i-1] for i in path[1:]]
