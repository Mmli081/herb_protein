import numpy as np
import networkx as nx
import random

def katz_index_similarity(graph, beta):
    adjacency_matrix = nx.to_numpy_array(graph)
    identity_matrix = np.identity(adjacency_matrix.shape[0])
    katz_matrix = np.linalg.pinv(identity_matrix - beta * adjacency_matrix) - identity_matrix

    return katz_matrix

def calc_KI(g, beta):
    # Calculate the adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(g).todense()

    # Calculate the Katz matrix
    katz_matrix = np.linalg.inv(np.eye(g.number_of_nodes()) - beta * adjacency_matrix) - np.eye(g.number_of_nodes())

    return katz_matrix


def calc_GLHN(g, beta1, beta2):
    # Calculate the adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(g).todense()

    # Calculate the GLHN matrix
    glhn_matrix = np.power(adjacency_matrix, beta1)
    glhn_matrix = glhn_matrix / np.sum(glhn_matrix, axis=0)
    glhn_matrix = glhn_matrix * np.power(adjacency_matrix, beta2)

    return glhn_matrix

def calc_simrank(g, node1, node2):
    return nx.simrank_similarity(g, node1, node2)

def calculate_plm_similarity(graph, node1, node2):
    laplacian_matrix = nx.laplacian_matrix(graph).todense()
    
    # Calculate pseudo-inverse of Laplacian matrices
    pseudo_inv_laplacian = np.linalg.pinv(laplacian_matrix)
    
    # Extract submatrices for the nodes and the edge between them
    submatrix_xy = pseudo_inv_laplacian[node1, node2]
    submatrix_xx = pseudo_inv_laplacian[node1, node1]
    submatrix_yy = pseudo_inv_laplacian[node2, node2]
    
    # Calculate PLM similarity
    plm_similarity = submatrix_xy / np.sqrt(submatrix_xx * submatrix_yy)
    
    return plm_similarity.item()


def calculate_hitting_time(graph, node_x, node_y):
    adjacency_matrix = nx.adjacency_matrix(graph).todense()
    degree_matrix_inv = np.linalg.inv(np.diag(np.sum(adjacency_matrix, axis=1)))
    
    # Calculate hitting time using the random walk assumption
    hitting_time = np.zeros_like(adjacency_matrix)
    np.fill_diagonal(hitting_time, 1)
    
    while not np.allclose(hitting_time[:, node_y], np.dot(degree_matrix_inv, np.dot(adjacency_matrix, hitting_time[:, node_y]))):
        hitting_time[:, node_y] = np.dot(degree_matrix_inv, np.dot(adjacency_matrix, hitting_time[:, node_y]))
    
    return hitting_time[node_x, node_y]

def calculate_average_commute_time(graph, node_x, node_y):
    hitting_time_x_y = calculate_hitting_time(graph, node_x, node_y)
    hitting_time_y_x = calculate_hitting_time(graph, node_y, node_x)
    
    # Calculate average commute time
    average_commute_time = hitting_time_x_y + hitting_time_y_x
    
    return average_commute_time


def calculate_rooted_pagerank(graph, node_x, node_y, beta=0.85):
    # Create the transition probability matrix
    transition_matrix = nx.pagerank_matrix(graph, alpha=beta).toarray()

    # Calculate the Rooted PageRank
    rpr_similarity = (1 - beta) * np.linalg.inv(np.eye(graph.number_of_nodes()) - beta * transition_matrix)

    return rpr_similarity[node_x, node_y]


def calculate_escape_probability(graph, node_x, node_y, beta):
    # Calculate SimRank matrix using NetworkX
    simrank_matrix = np.array(nx.simrank_similarity(graph))

    # Calculate Q(vx, vy)
    q_vx_vy = simrank_matrix[node_x][node_y]

    # Calculate Q(vx, vx) and Q(vy, vy)
    q_vx_vx = simrank_matrix[node_x][node_x]
    q_vy_vy = simrank_matrix[node_y][node_y]

    # Calculate Q(vx, vy) * Q(vy, vx)
    q_vx_vy_times_q_vy_vx = q_vx_vy * simrank_matrix[node_y][node_x]

    # Calculate Escape Probability (EP)
    escape_probability = q_vx_vy / (q_vx_vx * q_vy_vy - q_vx_vy_times_q_vy_vx)

    return escape_probability


def random_walk_with_restart(graph, node_x, node_y, restart_prob, max_iter=100, tol=1e-6):
    # Initialize probability vectors
    p_vx = np.zeros(len(graph.nodes))
    p_vx[node_x] = 1  # Start the random walk from node_x

    p_vy = np.zeros(len(graph.nodes))
    p_vy[node_y] = 1  # Start the random walk from node_y

    # Transition matrix M
    M = np.zeros((len(graph.nodes), len(graph.nodes)))
    for i in range(len(graph.nodes)):
        neighbors_sum = sum(graph[node_i][i].get('weight', 1) for node_i in graph.neighbors(i))
        for j in range(len(graph.nodes)):
            if neighbors_sum > 0:
                M[i, j] = graph[i][j].get('weight', 1) / neighbors_sum

    # Perform Random Walk with Restart
    for _ in range(max_iter):
        p_vx_new = np.dot(M.T, p_vx)
        p_vy_new = np.dot(M.T, p_vy)

        # Apply restart probability
        p_vx_new = restart_prob * p_vx_new + (1 - restart_prob) * np.identity(len(graph.nodes))[node_x]
        p_vy_new = restart_prob * p_vy_new + (1 - restart_prob) * np.identity(len(graph.nodes))[node_y]

        # Check convergence
        if np.linalg.norm(p_vx_new - p_vx, ord=1) < tol and np.linalg.norm(p_vy_new - p_vy, ord=1) < tol:
            break

        p_vx = p_vx_new
        p_vy = p_vy_new

    # Calculate RWR similarity
    rwr_similarity = p_vx[node_y] + p_vy[node_x]

    return rwr_similarity

def maximal_entropy_random_walk(graph, node_x, node_y, num_steps=10000):
    current_node = node_x
    visit_counts = {node: 0 for node in graph.nodes}

    for _ in range(num_steps):
        neighbors = list(graph.neighbors(current_node))
        if not neighbors:
            # If the current node has no neighbors, restart from the initial node
            current_node = node_x
        else:
            # Choose a random neighbor to move to
            next_node = random.choice(neighbors)
            visit_counts[next_node] += 1
            current_node = next_node

    # Calculate the empirical probabilities based on visit counts
    empirical_probabilities = {node: count / num_steps for node, count in visit_counts.items()}

    # Calculate the entropy
    entropy = -sum(prob * np.log(prob) for prob in empirical_probabilities.values() if prob > 0)

    return entropy

def calculate_blondel_similarity(A, max_iter=10):
    m, n = A.shape
    S = np.eye(m)  # Initialize S(0) as the identity matrix
    M_norm = np.linalg.norm(A, 'fro')  # Frobenius norm of the adjacency matrix A

    for t in range(1, max_iter + 1):
        update_term = (A @ S @ A.T + A.T @ S @ A) / ((np.linalg.norm(A @ S @ A.T + A.T @ S @ A, 'fro'))**2)

        S = update_term

    return S
