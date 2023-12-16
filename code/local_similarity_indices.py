import math, numpy as np
import networkx as nx

def nodes_connected(g, u, v):
    return u in g.neighbors(v)

def calc_CN(g, node1, node2, **kw):
    # Common Neighbors
    return len(list(nx.common_neighbors(g, node1, node2,)))

def calc_JC(g, node1, node2, **kw):
    # Jaccard index
    return next(nx.jaccard_coefficient(g, [(node1, node2)]))[2]

def calc_SL(g, node1, node2, **kw):
    degree_product = kw['d1'] * kw['d2']
    
    if degree_product == 0:
        return 0
    else:
        return len(kw['cn']) / (degree_product**0.5)
    

def calc_SI(g, node1, node2, **kw):
    total_degree = kw['d1'] + kw['d2']

    if total_degree == 0:
        return 0
    else:
        return len(kw['cn']) / total_degree
    
def calc_PA(g, node1, node2, **kw):
    #Preferential Attachment Similarity
    degree1 = kw['d1']
    degree2 = kw['d2']
    return degree1 * degree2

import math

def calc_AA(g, node1, node2, **kw):
    #Adamic adar index
    aa_similarity = 0
    for common_neighbor in kw['cn']:
        degree_common_neighbor = g.degree[common_neighbor]
        if degree_common_neighbor > 1:  # Avoid log(0) division
            aa_similarity += 1 / math.log(degree_common_neighbor)

    return aa_similarity

def calc_RA(g, node1, node2, **kw):
    #Resource_allocation_index
    ra_similarity = 0
    for common_neighbor in kw['cn']:
        degree_common_neighbor = g.degree[common_neighbor]
        if degree_common_neighbor > 0:  # Avoid division by zero
            ra_similarity += 1 / degree_common_neighbor

    return ra_similarity

def calc_HP(g, node1, node2, **kw):
    #hub_promoted_index
    if not kw['cn']:
        return 0
    
    hp_similarity = len(kw['cn']) / min(kw['d1'], kw['d2'])

    return hp_similarity

def calc_HD(g, node1, node2, **kw):
    #hub_depressed_index
    if not kw['cn']:
        return 0
    
    hd_similarity = len(kw['cn']) / max(kw['d1'], kw['d2'])

    return hd_similarity

def calc_LHN(g, node1, node2, **kw):
    #leicht_holme_newman_index
    if not kw['cn']:
        return 0
    
    lhn_similarity = len(kw['cn']) / (kw['d1'] * kw['d2'])

    return lhn_similarity

def calc_PD(g, node1, node2,beta_prime=2, **kw):
    #parameter_dependent_index
    #Beta with 0, 0.5, 1 equals SL, CN, LHN
    if not kw['cn']:
        return 0
    
    pd_similarity = len(kw['cn']) / (kw['d1'] * kw['d2']) ** beta_prime

    return pd_similarity

def calc_LAS(g, node1, node2, **kw):
    #local_attachment_strength
    if not kw['cn']:
        return 0
    
    las_similarity = len(kw['cn']) / kw['d1'] + len(kw['cn']) / kw['d2']

    return las_similarity

def calc_CB(g, node1, node2, **kw):
    #car_based_index
    if not kw['cn']:
        return 0
    
    car_similarity = len(kw['cn']) / 2 * sum(g.degree[vz] for vz in kw['cn'])

    return car_similarity

def calc_IA(g, node1, node2, **kw):
    #individual_attraction_index
    ia_similarity = 0.0

    for vz in kw['cn']:
        common_neighbors_vz = set(g.neighbors(vz)).intersection(kw['cn'])

        # Calculate Individual Attraction Index similarity
        ia_similarity += (len(common_neighbors_vz) + 2) / len(list(g.neighbors(vz)))

    return ia_similarity

def calc_MI(g, node1, node2, **kw):
    #mutual_information_index
    if not kw['cn']:
        return 0

    intersec = set(g.edges(kw['neighbors1'])).union(set(g.edges(kw['neighbors2'])))

    mi = 0
    for vz in kw['cn']:
        common_edges = set(g.edges(vz)).intersection(intersec)
        mi -= math.log2(len(common_edges) / (0.5 * len(kw['neighbors1']) * (len(kw['neighbors1']) - 1)))

    return mi

def calc_FSW(g, node1, node2, avg_neighbors, **kw):
    #functional_similarity_weight
    if not kw['cn']:
        return 0
    
    beta = max(0, avg_neighbors - len(kw['neighbors1']) + len(kw['neighbors2']) + len(kw['cn']))
    fsw_similarity = (2 * len(kw['cn']) / (len(kw['neighbors1']) - len(kw['neighbors2']) + 2 * len(kw['cn']) + beta))**2

    return fsw_similarity

def calc_LNL(g, node1, node2, **kw):
    #local_neighbors_link_index
    if not kw['cn']:
        return 0
    
    lnl_similarity = 0
    for vz in kw['cn']:
        w_vz = (sum(1 for vu in kw['neighbors1'].union({node1}) if g.has_edge(vz, vu)) +
                sum(1 for vv in kw['neighbors2'].union({node2}) if g.has_edge(vz, vv))) / g.degree(vz)
        lnl_similarity += w_vz

    return lnl_similarity


features = [calc_CN,calc_JC,calc_SL,calc_SI,calc_PA,calc_AA,calc_RA,calc_HP,\
            calc_HD,calc_LHN,calc_PD,calc_LAS,calc_CB,calc_IA,calc_MI,calc_FSW,\
            calc_LNL]


def calc_local_similarity(g, node1, node2, avg_neighbors):
    neighbors1 = set(g.neighbors(node1))
    neighbors2 = set(g.neighbors(node2))

    # Calculate intersection of neighbor sets
    common_neighbors = neighbors1.intersection(neighbors2)

    # Calculate degree of nodes
    degree1 = g.degree[node1]
    degree2 = g.degree[node2]

    return {"node1":node1, "node2":node2, "d1":degree1,\
             "d2":degree2, "cn":common_neighbors, 'g':g,\
             "neighbors1":neighbors1, "neighbors2":neighbors2,\
             "avg_neighbors":avg_neighbors  }
