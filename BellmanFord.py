import time

# global state populated by bellman_ford_algorithm
_runtime_seconds = 0.0
_all_paths = {}  # maps each reachable node to its full path from source
_all_costs = {}  # maps each reachable node to its total path cost

def bellman_ford_algorithm(graph, source, target=None):
    """Run Bellman-Ford shortest-path algorithm.

    graph  -- adjacency dict: {node: [(neighbor, weight), ...]}
    source -- starting node
    target -- optional destination node; if None, compute paths to all vertices

    returns the shortest path (list of nodes) to target, or a dict of all
    shortest paths keyed by destination node when target is None.
    """
    global _runtime_seconds, _all_paths, _all_costs

    # initialize distances to infinity for all nodes except source
    dist = {node: float('inf') for node in graph}
    if source in dist:
        dist[source] = 0

    # track predecessor of each node to reconstruct paths
    prev = {node: None for node in graph}

    start_time = time.perf_counter()

    # Relax all edges V - 1 times
    num_vertices = len(graph)
    for _ in range(num_vertices - 1):
        updated = False
        for current_node in graph:
            if dist.get(current_node, float('inf')) != float('inf'):
                for neighbor, weight in graph.get(current_node, []):
                    # Ensure neighbor exists in dist dictionary
                    if neighbor not in dist:
                        dist[neighbor] = float('inf')
                        prev[neighbor] = None
                        
                    new_dist = dist[current_node] + weight
                    if new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
                        prev[neighbor] = current_node
                        updated = True
        
        # Early exit if no distances were updated in this pass
        if not updated:
            break

    # Check for negative-weight cycles (standard Bellman-Ford step)
    for current_node in graph:
        if dist.get(current_node, float('inf')) != float('inf'):
            for neighbor, weight in graph.get(current_node, []):
                if dist[current_node] + weight < dist[neighbor]:
                    raise ValueError("Graph contains a negative-weight cycle.")

    _runtime_seconds = time.perf_counter() - start_time

    # reconstruct path for a single node
    def _reconstruct(node):
        path = []
        while node is not None:
            path.append(node)
            node = prev.get(node)
        path.reverse()
        # return empty list if node was unreachable
        return path if path and path[0] == source else []

    if target is not None:
        path = _reconstruct(target)
        _all_paths = {target: path}
        _all_costs = {target: dist.get(target, float('inf'))}
        return path

    # build paths for every reachable node
    _all_paths = {}
    _all_costs = {}
    for node in dist:
        if dist[node] < float('inf'):
            _all_paths[node] = _reconstruct(node)
            _all_costs[node] = dist[node]

    return _all_paths

def print_bellman_ford_analytics():
    """Print runtime and all paths recorded by the last algorithm run."""
    print(f"runtime: {_runtime_seconds:.6f} seconds")
    print(f"paths found: {len(_all_paths)}")
    for destination, path in sorted(_all_paths.items(), key=lambda x: str(x[0])):
        path_str = ' → '.join(str(n) for n in path) if path else 'unreachable'
        cost = _all_costs.get(destination)
        cost_str = f" (Cost: {cost})" if cost is not None else ''
        print(f"  {cost_str} Target: {destination}: {path_str}")
