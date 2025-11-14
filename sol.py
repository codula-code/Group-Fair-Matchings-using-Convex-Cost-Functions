#usage : python3 sol.py < top_<10,20,50,75,100>

from ortools.graph.python import min_cost_flow
from ortools.linear_solver import pywraplp
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict

def lp(graph, source, sink, l, uu):
    solver = pywraplp.Solver.CreateSolver('GLOP')
    all_nodes = set()
    edge_indices = {}
    
    for u in graph:
        all_nodes.add(u)
        for v, _, _ in graph[u]:
            all_nodes.add(v)
    flow_vars = {}
    for u in graph:
        for i, (v, capacity, cost) in enumerate(graph[u]):
            edge_id = i
            flow_vars[(u, v, edge_id)] = solver.NumVar(0, capacity, f'flow_{u}_{v}_{edge_id}')
    objective = solver.Objective()
    for u in graph:
        for i, (v, capacity, cost) in enumerate(graph[u]):
            edge_id = i
            objective.SetCoefficient(flow_vars[(u, v, edge_id)], cost)
    objective.SetMinimization()
    for node in all_nodes:
        if node == source or node == sink:
            continue
        constraint = solver.Constraint(0, 0)
        for u in graph:
            for i, (v, _, _) in enumerate(graph[u]):
                if v == node:
                    edge_id = i
                    constraint.SetCoefficient(flow_vars[(u, v, edge_id)], 1)
        for i, (v, _, _) in enumerate(graph[node]):
            edge_id = i
            constraint.SetCoefficient(flow_vars[(node, v, edge_id)], -1)
    uti_constraint = solver.Constraint(l, solver.infinity())
    for u in graph:
        for i, (v, capacity, cost) in enumerate(graph[u]):
            edge_id = i
            if u[0] == 'i':
                uti_constraint.SetCoefficient(flow_vars[(u, v, edge_id)], uu[(u, v.split('_')[0])])
    

    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        flow_solution = {}
        total_cost = 0
        for u in graph:
            for i, (v, _, cost) in enumerate(graph[u]):
                edge_id = i
                flow_val = flow_vars[(u, v, edge_id)].solution_value()
                flow_solution[(u, v, edge_id)] = flow_val
                total_cost += flow_val * cost
        return flow_solution
    elif status == pywraplp.Solver.FEASIBLE:
        return None
    else:
        assert False
        return None

from collections import deque

def find_fractional_paths(graph, flow, num_items, eps=1e-10):
    frac_edges = []
    for (u, v, eid), f in flow.items():
        if u.startswith('i') and eps < f < 1-eps:
            frac_edges.append((u, v, eid))
    if len(frac_edges) > 2:
        raise RuntimeError(f"Found {len(frac_edges)} fractional item→platform edges")

    paths = []
    for (ui, v0, e0) in frac_edges:
        parent = {}
        q = deque([v0])
        found = False
        while q and not found:
            u = q.popleft()
            for eid, (v, _, _) in enumerate(graph.get(u, [])):
                edge = (u, v, eid)
                if edge not in parent and flow[edge] > 0:
                    parent[v] = (u, eid)
                    if v == 't':
                        found = True
                        break
                    q.append(v)
        suffix = []
        cur = 't'
        while cur != v0:
            prev, eid = parent[cur]
            suffix.append((prev, cur, eid))
            cur = prev
        suffix.reverse()
        for eid_s, (v1, _, _) in enumerate(graph['s']):
            if v1 == ui:
                prefix = [('s', ui, eid_s), (ui, v0, e0)]
                break
        else:
            raise RuntimeError(f"Could not find s→{ui} edge")

        full_path = prefix + suffix
        paths.append(full_path)

    return paths

def rounding_while_plotting(graph, flow, num_items, uu,eps=1e-6):
    from ortools.graph.python import min_cost_flow
    from ortools.linear_solver import pywraplp
    import math
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from collections import defaultdict

    frac_paths = find_fractional_paths(graph, flow, num_items)
    if len(frac_paths) > 2:
        raise RuntimeError(f"Expected <=2 fractional paths, got {len(frac_paths)}")

    total_orig = sum(
        f * uu[(u, v.split('_')[0])]
        for (u, v, eid), f in flow.items()
        if u.startswith('i')
    )
    rounded_flow = flow.copy()
    if len(frac_paths) == 2:
        utilities = []
        for path in frac_paths:
            utilities.append(sum(uu[(u,v.split('_')[0])] for (u,v,_) in path if u.startswith('i')))
        idx_up = 0 if utilities[0] >= utilities[1] else 1
        idx_down = 1 - idx_up
    elif len(frac_paths) == 1:
        idx_up, idx_down = 0, None
    else:
        idx_up = idx_down = None

    for pi, path in enumerate(frac_paths):
        target = 1.0 if pi == idx_up else 0.0
        for edge in path:
            if target==1:
                rounded_flow[edge]=math.ceil(rounded_flow[edge])
            else:
                rounded_flow[edge]=math.floor(rounded_flow[edge])

    total_round = sum(
        f * uu[(u, v.split('_')[0])]
        for (u, v, eid), f in rounded_flow.items()
        if u.startswith('i')
    )
    round_cost = 0
    for u in graph:
        for i,(v,_,cost) in enumerate(graph[u]):
            round_cost+=rounded_flow[(u,v,i)]*cost


    costt = 0
    hw_many = defaultdict(int)
    hw_many2 = defaultdict(lambda: defaultdict(int))
    for u in graph:
        for i,(v,_,cost) in enumerate(graph[u]):
            costt+=flow[(u,v,i)]*cost
            if(v[0]=='P' and v[1]=='2' and rounded_flow[(u,v,i)]>1-(1e-10)):
                hw_many[v[3:]]+=1
            if(v[0]=='P' and v[1]=='1' and rounded_flow[(u,v,i)]>1-(1e-10)):
                hw_many2[v[3:]][u.split('_')[1]]+=1
    
    print("Before rounding utility -: ",total_orig)
    print("After rounding utility -: ",total_round)
    print("After rounding cost -: ",round_cost)
    print("Before rounding cost -: ",costt)

    category_matches = hw_many2

    match_counts = hw_many
    plt.figure(figsize=(10, 6))
    keys = list(match_counts.keys())
    counts = list(match_counts.values())

    plt.subplot(3,1,3)
    df = pd.DataFrame(category_matches).T

    hatches = ['', '////', '\\\\\\\\', 'xxxx', '....', 'OO']
    grays = ['white', '#DDDDDD', '#BBBBBB', '#999999', '#777777', '#555555']

    ax = df.plot(kind='bar', stacked=True, ax=plt.gca(), 
                color=grays, edgecolor='black')

    for i, bar in enumerate(ax.containers):
        for j, patch in enumerate(bar.patches):
            patch.set_hatch(hatches[i % len(hatches)])

    plt.xlabel('Platforms', fontsize=12)
    plt.ylabel('Number of Items matched', fontsize=12)
    plt.title('Groupwise Distribution', fontsize=14, fontweight='bold')

    plt.legend(title='Groups', title_fontsize=12, fontsize=10, 
              loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.gcf().set_size_inches(10, 10)
    plt.tight_layout()
    plt.show()

    plt.subplot(3, 1, 2)
    plt.scatter(keys, counts, color='red', s=100, zorder=2)
    plt.plot(keys, counts, 'b--', zorder=1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('platform')
    plt.ylabel('Number of Matched Elements')
    plt.title('Number of Elements Matched per platform')

    plt.subplot(3, 1, 2)
    plt.figure(figsize=(10, 4))
    plt.plot(keys, counts, marker='o')
    plt.yticks(range(min(counts)-5, max(counts)+5, 1))
    plt.xticks(keys)
    plt.grid(True)
    plt.xlabel('Platform')
    plt.ylabel('Number of Matched items')
    plt.title('Top_10')

    return costt

def rounding(graph, flow, num_items, uu, eps=1e-6):

    frac_paths = find_fractional_paths(graph, flow, num_items)
    if len(frac_paths) > 2:
        # print(frac_paths)
        raise RuntimeError(f"Expected ≤2 fractional paths, got {len(frac_paths)}")

    total_orig = sum(
        f * uu[(u, v.split('_')[0])]
        for (u, v, eid), f in flow.items()
        if u.startswith('i')
    )
    rounded_flow = flow.copy()
    if len(frac_paths) == 2:
        utilities = []
        for path in frac_paths:
            utilities.append(sum(uu[(u,v.split('_')[0])] for (u,v,_) in path if u.startswith('i')))
        idx_up = 0 if utilities[0] >= utilities[1] else 1
        idx_down = 1 - idx_up
    elif len(frac_paths) == 1:
        idx_up, idx_down = 0, None
    else:
        idx_up = idx_down = None

    for pi, path in enumerate(frac_paths):
        target = 1.0 if pi == idx_up else 0.0
        for edge in path:
            if target==1:
                rounded_flow[edge]=math.ceil(rounded_flow[edge])
            else:
                rounded_flow[edge]=math.floor(rounded_flow[edge])

    total_round = sum(
        f * uu[(u, v.split('_')[0])]
        for (u, v, eid), f in rounded_flow.items()
        if u.startswith('i')
    )
    round_cost = 0
    for u in graph:
        for i,(v,_,cost) in enumerate(graph[u]):
            round_cost+=rounded_flow[(u,v,i)]*cost
    

    costt = 0
    hw_many = defaultdict(int) 
    hw_many2 = defaultdict(lambda: defaultdict(int))
    for u in graph:
        for i,(v,_,cost) in enumerate(graph[u]):
            costt+=flow[(u,v,i)]*cost
            if(v[0]=='P' and v[1]=='2' and rounded_flow[(u,v,i)]>1-(1e-10)):
                hw_many[v[3:]]+=1
            if(v[0]=='P' and v[1]=='1' and rounded_flow[(u,v,i)]>1-(1e-10)):
                hw_many2[v[3:]][u.split('_')[1]]+=1  
                
    print("Before rounding utility -: ",total_orig)
    print("After rounding utility -: ",total_round)
    print("After rounding cost -: ",round_cost)
    print("Before rounding cost -: ",costt)

    return costt


def add_edge(graph, u, v, capacity, cost):
    if u not in graph:
        graph[u] = []
    if v not in graph:
        graph[v] = []
    graph[u].append((v, capacity, cost))

def power(a,b):
    
    if b==0:
        return 1
    
    out_ = power(a,b//2)
    out_ = out_ * out_
    
    if b%2 !=0 :
        out_ = out_ * a
    
    return out_

import math

def f(k):
    x=k
    if(k==0):
        return 0
    else:
        return x**2

def g(k):
    x=k
    if(k==0):
        return 0
    else:
        return 2*x**2 + 3*x


def build_flow_network(G, I_groups, tau, num_platforms, num_items):
    sz = 0
    graph = {}
    source = "s"
    sink = "t"

    count1 = {}
    count2 = {}
    for i in range(1,num_items+1):
        add_edge(graph, source, f"i{i}", capacity=1, cost=0)
        sz+=1

    for i in range(1,num_platforms+1):
      for j in range(tau):
        if f"p{i}_{j}" not in count1:
            count1[f"p{i}_{j}"]=0
        if f"p{i}" not in count2:
            count2[f"p{i}"]=0

    for group_idx, group in enumerate(I_groups):
        for i in group:
            for j in range(1,num_platforms+1):
                if f"p{j}_{group_idx}" not in count1:
                    count1[f"p{j}_{group_idx}"]=0
                if f"p{j}" not in count2:
                    count2[f"p{j}"]=0
                if (i,f"p{j}") in G:
                    add_edge(graph,i,f"p{j}_{group_idx}",capacity=1,cost=0)
                    sz+=1
                    count1[f"p{j}_{group_idx}"]+=1
                    count2[f"p{j}"]+=1

    coeff = 1
    for p in range(1, num_platforms + 1):
        P1 = f"P1_{p}"
        for t in range(tau):
            for k in range(count1[f"p{p}_{t}"]):
                add_edge(graph,f"p{p}_{t}",P1,capacity=1,cost = (g(k+1)-g(k)))
                sz+=1
            # coeff+=1

    for p in range(1, num_platforms + 1):
        P1 = f"P1_{p}"
        P2 = f"P2_{p}"
        for k in range(count2[f"p{p}"]):
            add_edge(graph, P1, P2, capacity=1, cost=(f(k+1)-f(k)))
            sz+=1
        # coeff+=1

    for p in range(1, num_platforms + 1):
        P2 = f"P2_{p}"
        add_edge(graph, P2, sink, capacity=num_items, cost=0)
        sz+=1

    return graph


def display_network(graph):
    print("Flow Network:")
    for u in graph:
        for v, capacity, cost in graph[u]:
            print(f"{u} -> {v} | Capacity: {capacity}, Cost: {cost}")


def hashed(graph):
    m=0
    index = {}
    rev_index = {}
    for i in graph:
        if i not in index:
            index[i]=m
            rev_index[m]=i
            m+=1
        for j,c1,c2 in graph[i]:
            if j not in index:
                index[j]=m
                rev_index[m]=j
                m+=1
    return index,rev_index

def numb(s):
    h=""
    h=s.split('_')[1]
    return int(h)


def find_flow(original,graph,num_platforms,num_items,index,rev_index,which_group):
    flow = min_cost_flow.SimpleMinCostFlow()
    start=[]
    end=[]
    capacity=[]
    cost=[]
    for i in graph:
        for j,c1,c2 in graph[i]:
            start.append(index[i])
            end.append(index[j])
            capacity.append(c1)
            cost.append(c2)
    supply=[0]*(len(index)+1)
    supply[index['s']]=num_items
    supply[index['t']]=-num_items

    for i in range(len(start)):
        arc=flow.add_arc_with_capacity_and_unit_cost(start[i],end[i],capacity[i],cost[i])
        if arc!=i:
            print("CRASH!!")

    nodess = []
    for i in original:
        nodess.append(int(i[0][1:]))
    nodess=list(set(nodess))
    for i in range(len(supply)):
        flow.set_node_supply(i,supply[i])
    flow.solve()
    print(flow.optimal_cost())
    return flow.optimal_cost()


from collections import defaultdict
import math

def greedy(edges, group_of, L,uu):
    user_edges = defaultdict(list)
    for u, m in edges:
        user_edges[u].append((m,uu[(u,m)]))
    ans = 10000000000000000
    match = []
    uti = 0.0
    assigned   = {}                # user -> movie
    x_total    = defaultdict(int)  # movie -> total assigned
    x_group    = defaultdict(lambda: defaultdict(int))  # movie -> group -> count
    R          = 0                 # accumulated rating
    C          = 0.0               # accumulated convex cost
    matching   = []                # list of (user, movie, rating)

    def delta_cost(u, m):
        xi = x_total[m]
        gi = x_group[m][group_of[int(u[1:])]]
        df = f(xi+1) - f(xi)
        dg = g(gi+1) - g(gi)
        return df + dg

    while R < L:
        best = None
        best_rho = -math.inf
        for u, pairs in user_edges.items():
            if u in assigned:
                continue
            for m, r in pairs:
                if r <= 0:
                    continue
                dC = delta_cost(u, m)
                rho = math.inf if dC == 0 else r / dC
                if rho > best_rho:
                    best_rho = rho
                    best = (u, m, r, dC)

        if best is None or best_rho <= 0:
            print(f"Stopped early: total rating = {R}, total cost = {C}")
            break

        u_star, m_star, r_star, dC_star = best
        assigned[u_star] = m_star
        x_total[m_star] += 1
        x_group[m_star][group_of[int(u_star[1:])]] += 1
        R += r_star
        C += dC_star
        matching.append((u_star, m_star, r_star))

    return C

def naive_greedy(num_items,num_platforms,num_groups,edges, budget,which_group,uu):
    edges_sorted = sorted(edges, key=lambda e: int(uu[(e[0],e[1])]), reverse=True)

    selected = []
    total = 0
    plat = {}
    plat_grp = {} 
    for i in range(num_platforms):
        p = f"p{i+1}"
        plat[p] = []
        plat_grp[f"p{i+1}_{0}"] = []
        for j in range(num_groups):
            plat_grp[f"p{i+1}_{j+1}"] = []
    i = 0
    used = {}
    while total<budget:
        (u,v) = edges_sorted[i]
        if used.get(u)==None:
            used[u]=1
            w = uu[(u,v)]
            selected.append((u, v, w))
            grp = which_group[int(u[1:])]
            plat[v].append(u)
            plat_grp[f"{v}_{grp}"].append(u)
            total += w
        i+=1

    for (i,j,k) in selected:
        assert used[i]<=1

    cost = 0.00
    for i in plat.keys():
        cost += 1.00*f(len(plat[i]))
    for i in plat_grp.keys():
        cost += 1.00*g(len(plat_grp[i]))

    print("naive greedy total utility : ",total)
    print("naive greedy total cost : ",cost)

def take_input():
    n=int(input())
    m=int(input())
    k=int(input())
    l = int(input())

    summ = 0

    numbering = {}
    last = 1
    I_groups = []
    uu = {}
    which_group = {}
    mny={0:0,1:0,2:0,3:0,4:0,5:0}
    for i in range(k):
        I_groups.append([])
    graph = {}
    for i in range(1,n+1):
        h = int(input())
        for j in range(h):
            tt = int(input())
            if numbering.get(tt) == None:
                numbering[tt] = last
                tt = last
                last+=1
            else:
                tt = numbering[tt]
            uti = int((input()).split('.')[0])
            graph[(f"i{i}",f"p{tt}")] = True
            uu[(f"i{i}",f"p{tt}")] = uti
            summ+=1
            mny[uti]+=1

    test = 0
    for i in range(n):
        g = int(input())
        which_group[i+1]=g
        I_groups[g-1].append(f"i{i+1}")

    return n,m,k,graph,I_groups,which_group,uu,l

def input_distribution(I_grp, grp):
    platforms = set()
    for edge in grp.keys():
        if grp[edge]:
            item, platform = edge
            platforms.add(platform)

    item_to_group = {}
    for group_idx, items in enumerate(I_grp):
        for item in items:
            item_to_group[item] = group_idx
    distribution = {}
    for platform in platforms:
        group_counts = {i: 0 for i in range(len(I_grp))}
        
        for edge in grp.keys():
            if grp[edge]: 
                item, plat = edge
                if plat == platform:
                    group = item_to_group.get(item, None)
                    if group is not None:
                        group_counts[group] += 1
        
        distribution[platform] = group_counts
    return distribution


if __name__ == "__main__":
    import math
    num_items,num_platforms,tau,G,I_groups,which_group,uu,l = take_input()
    naive_greedy(num_items,num_platforms,tau,G,l,which_group,uu)
    greed=greedy(G,which_group,l,uu)
    print("Smart greedy : ",greed)
    network = build_flow_network(G, I_groups, tau, num_platforms, num_items)
    OPT=rounding(network,lp(network,'s','t',l,uu),num_items,uu)