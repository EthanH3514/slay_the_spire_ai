"""Inference helpers from escape_distribution.py."""

import math
from combat_resource_utility import scalar, validate_version, LEGACY, LINEAR
def summarize_distributions(rows):
    """Python oracle: weighted inverse CDF and exact fractional lower-tail mass."""
    if not rows:raise ValueError('empty distribution')
    atoms=[];loss=win=resources=values=0.
    for row in rows:
        version=validate_version(row.get('resource_utility',LEGACY))
        if row['source']=='terminal':
            if version==LINEAR:
                expected=scalar(row['outcome'],row['resource_hp']+row['root_reserve'],row['max_hp'],version)
                if not math.isclose(expected,row['value'],rel_tol=1e-12,abs_tol=1e-10):
                    raise ValueError('terminal resource utility mismatch')
            a=[(1.,row['resource_hp'],row['value'],row['outcome'])]
        elif row['source']=='model':
            p=row['probabilities'];q=row['resource_quantiles'];reserve=row['root_reserve'];hp=row['max_hp']
            if len(p)!=3 or len(q)!=3 or any(len(x)!=21 for x in q) or hp<=0 or reserve<0:raise ValueError('value shape')
            if any(not math.isfinite(x) or x<0 for x in p) or abs(sum(p)-1)>1e-8:raise ValueError('value mass')
            if any(any(not math.isfinite(x) or x<0 for x in branch) or branch!=sorted(branch) for branch in q):raise ValueError('value quantiles')
            a=[(p[k]/21,x*100-reserve,scalar([2,1,3][k],x*100,hp,version),[2,1,3][k]) for k in range(3) for x in q[k]]
        else:raise ValueError('unknown distribution source')
        if abs(sum(x[0] for x in a)-1)>1e-8:raise ValueError('distribution mass')
        for mass,resource,value,outcome in a:
            if outcome not in (1,2,3) or any(not math.isfinite(x) for x in (mass,resource,value)) or mass<0:raise ValueError('distribution atom')
            if mass==0:continue
            atoms.append((resource,mass));loss+=mass*(outcome==2);win+=mass*(outcome==1)
            resources+=mass*resource;values+=mass*value
    n=len(rows);atoms.sort();remaining=.2*n;lower=0.;cumulative=0.;p10=None
    for resource,mass in atoms:
        part=min(remaining,mass);lower+=part*resource;remaining-=part
        cumulative+=mass
        if p10 is None and cumulative>=.1*n-1e-12:p10=resource
    if remaining>1e-8 or p10 is None:raise ValueError('missing tail mass')
    return [-loss/n,win/n,lower/(.2*n),p10,resources/n,values/n]
