from dataclasses import dataclass
import heapq
import math
from typing import List, Tuple, Iterable, Optional, Dict, Set

INF = float("inf")

@dataclass
class Edge:
    u: int
    v: int
    w: float

class Graph:
    def __init__(self, n: int):
        self.n = n
        self.adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        self.rev_adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]  # may help in debugging

    def add_edge(self, u: int, v: int, w: float) -> None:
        assert w >= 0, "Nonnegative weights required"
        self.adj[u].append((v, w))
        self.rev_adj[v].append((u, w))

class PriorityBatch:
    """
    A simple replacement for the data structure `D` in Lemma 3.3.

    - insert(key, item): amortized O(log N) here (heap push)
    - batch_prepend(list_of_pairs): we simply push all -> O(k log N)
    - pull(): pop up to M smallest keys and return (S, B), where B is the last popped key (or +inf if none)

    This preserves the algorithmic *behavior* needed by BMSSP, but not the improved asymptotics.
    """
    def __init__(self, M: int):
        self.M = max(1, int(M))
        self._heap: List[Tuple[float, int]] = []  # (key, vertex)

    def __len__(self) -> int:
        return len(self._heap)

    def insert(self, key: float, item: int) -> None:
        heapq.heappush(self._heap, (key, item))

    def batch_prepend(self, pairs: Iterable[Tuple[float, int]]) -> None:
        for key, item in pairs:
            heapq.heappush(self._heap, (key, item))

    def pull(self) -> Tuple[Set[int], float]:
        S: Set[int] = set()
        last_key = INF
        cnt = 0
        while self._heap and cnt < self.M:
            key, item = heapq.heappop(self._heap)
            last_key = key
            S.add(item)
            cnt += 1
        if cnt == 0:
            return set(), INF
        return S, last_key

class SSSPBreaker:
    def __init__(self, g: Graph, source: int):
        self.g = g
        self.s = source
        n = g.n
        # db[·] in paper — always a *sound* upper bound of true distances, monotonically nonincreasing
        self.db: List[float] = [INF] * n
        self.db[self.s] = 0.0
        # predecessor tree
        self.pred: List[Optional[int]] = [None] * n
        # Track 'complete' vertices per the paper (those whose db == true shortest distance by construction)
        self.complete: Set[int] = {self.s}
        # Parameters: k = floor((log n)^(1/3)), t = floor((log n)^(2/3))
        # Use natural log; base does not matter for polylog expressions.
        logn = max(1.0, math.log(max(2, n)))
        self.k = max(1, int(logn ** (1/3)))
        self.t = max(1, int(logn ** (2/3)))

    # ---------- Utility: edge relaxation with <= condition ----------
    def relax(self, u: int, v: int, w: float, bound: float) -> bool:
        """Try relaxing (u,v); only keep if new value < bound. Equality is allowed in the <=
        test to permit reuse across levels, as in the paper (Remark 3.4). Returns True if db[v] changed."""
        nu = self.db[u] + w
        if nu <= self.db[v] and nu < bound:
            self.db[v] = nu
            self.pred[v] = u
            return True
        return False

    # ---------- Algorithm 2: Base case (mini Dijkstra from singleton S={x}) ----------
    def base_case(self, B: float, S: Set[int]) -> Tuple[float, Set[int]]:
        assert len(S) == 1
        (x,) = tuple(S)
        # U0 grows with discovered vertices in ascending db
        U0: Set[int] = {x}
        H: List[Tuple[float, int]] = [(self.db[x], x)]
        seen = {x}
        while H and len(U0) < self.k + 1:
            du, u = heapq.heappop(H)
            # Ignore stale heap entry
            if du != self.db[u]:
                continue
            U0.add(u)
            for v, w in self.g.adj[u]:
                old = self.db[v]
                if self.relax(u, v, w, B):
                    if v not in seen:
                        heapq.heappush(H, (self.db[v], v))
                        seen.add(v)
                    else:
                        heapq.heappush(H, (self.db[v], v))
        if len(U0) <= self.k:
            # Successful execution for base case
            return B, U0
        # Otherwise, set boundary B' to max db in U0, and return those below B'
        Bp = max(self.db[v] for v in U0)
        U = {v for v in U0 if self.db[v] < Bp}
        return Bp, U

    # ---------- Algorithm 1: FindPivots ----------
    def find_pivots(self, B: float, S: Set[int]) -> Tuple[Set[int], Set[int]]:
        W = set(S)
        W_prev = set(S)
        # Perform k relaxation rounds from S (bounded by B)
        for _ in range(self.k):
            Wi: Set[int] = set()
            for u in list(W_prev):
                for v, w in self.g.adj[u]:
                    # If <= and still under B add v to next frontier
                    if self.db[u] + w <= self.db[v]:
                        # Update db[v] even if equals to enable tie-breaking reuse across levels
                        if self.db[u] + w < self.db[v]:
                            self.db[v] = self.db[u] + w
                            self.pred[v] = u
                        if self.db[u] + w < B:
                            Wi.add(v)
            W |= Wi
            W_prev = Wi
        # If frontier exploded, use all S as pivots
        if len(W) > self.k * len(S):
            return set(S), W
        # Build directed forest F on W using tight edges (u,v) with db[v] == db[u] + w
        children: Dict[int, List[int]] = {u: [] for u in W}
        indeg: Dict[int, int] = {u: 0 for u in W}
        for u in W:
            for v, w in self.g.adj[u]:
                if v in W and math.isclose(self.db[v], self.db[u] + w, rel_tol=0.0, abs_tol=0.0):
                    children[u].append(v)
                    indeg[v] += 1
        # Identify roots and subtree sizes; pick roots with >= k nodes
        roots = [u for u in W if indeg.get(u, 0) == 0]
        size: Dict[int, int] = {}
        def dfs_sz(x: int) -> int:
            s = 1
            for y in children.get(x, []): s += dfs_sz(y)
            size[x] = s
            return s
        for r in roots: dfs_sz(r)
        P = {u for u in S if size.get(u, 0) >= self.k}
        return P, W

    # ---------- Algorithm 3: BMSSP ----------
    def bmssp(self, l: int, B: float, S: Set[int]) -> Tuple[float, Set[int]]:
        if l == 0:
            return self.base_case(B, S)
        P, W = self.find_pivots(B, S)
        M = 2 ** max(0, (l - 1) * self.t)
        D = PriorityBatch(M)
        for x in P: D.insert(self.db[x], x)
        i = 0
        B0p = min((self.db[x] for x in P), default=B)
        U: Set[int] = set()
        while len(U) < self.k * (2 ** (l * self.t)) and len(D) > 0:
            i += 1
            Si, Bi = D.pull()  # keys with smallest values, boundary lower bound
            if not Si:
                break
            # Recurse on each pulled group Si as a set
            Bip, Ui = self.bmssp(l - 1, Bi, set(Si))
            U |= Ui
            # Relax outgoing edges of Ui and feed neighbors back to D per rules
            K: List[Tuple[float, int]] = []
            for u in Ui:
                for v, w in self.g.adj[u]:
                    newval = self.db[u] + w
                    if newval <= self.db[v]:
                        # Tighten db[v] even if equal; we maintain pred on strict improv.
                        if newval < self.db[v]:
                            self.db[v] = newval
                            self.pred[v] = u
                        if Bip <= newval < B:
                            if newval >= Bi:
                                D.insert(newval, v)
                            else:
                                K.append((newval, v))
            # Also batch prepend keys for Si within [B', Bi)
            for x in Si:
                dx = self.db[x]
                if Bip <= dx < Bi:
                    K.append((dx, x))
            if K:
                D.batch_prepend(K)
        Bp = min([B0p, B] + [self.db[x] for x in W if self.db[x] < B])
        # Include vertices from W made complete below B'
        U |= {x for x in W if self.db[x] < Bp}
        return Bp, U

    # ---------- Driver ----------
    def run(self) -> Tuple[List[float], List[Optional[int]]]:
        n = self.g.n
        top_l = max(1, math.ceil(math.log(max(2, n)) / max(1, self.t)))  # ≈ ⌈(log n)/t⌉; keep >=1
        _B, U = self.bmssp(top_l, INF, {self.s})
        # Distances are in self.db; predecessors in self.pred
        return self.db, self.pred

# Public API

def sssp_break_sorting(g: Graph, source: int = 0) -> Tuple[List[float], List[Optional[int]]]:
    """Compute single-source shortest paths using the paper's BMSSP framework.

    Parameters
    ----------
    g : Graph
        Directed graph with nonnegative weights.
    source : int
        Source vertex index.

    Returns
    -------
    dist : List[float]
        Distance to each vertex (INF if unreachable).
    pred : List[Optional[int]]
        Predecessor pointers that define a shortest-path tree.
    """
    alg = SSSPBreaker(g, source)
    dist, pred = alg.run()
    return dist, pred

if __name__ == "__main__":
    g = Graph(5)
    g.add_edge(0, 1, 2)
    g.add_edge(0, 2, 5)
    g.add_edge(1, 2, 1)
    g.add_edge(1, 3, 2)
    g.add_edge(2, 3, 1)
    g.add_edge(3, 4, 3)
    dist, pred = sssp_break_sorting(g, 0)
    print("dist:", dist)
    print("pred:", pred)
