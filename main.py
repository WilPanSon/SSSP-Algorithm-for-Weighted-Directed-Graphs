from __future__ import annotations
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
        self.rev_adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, w: float) -> None:
        assert w >= 0, "Nonnegative weights required"
        self.adj[u].append((v, w))
        self.rev_adj[v].append((u, w))

class PriorityBatch:
    """Heap-backed batch queue. Simpler than Lemma 3.3, but preserves behavior."""
    def __init__(self, M: int):
        self.M = max(1, int(M))
        self._heap: List[Tuple[float, int]] = []

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
    def __init__(self, g: Graph, source: int,
                 max_depth: int = 2,
                 iter_cap_multiplier: int = 5,
                 relax_cap_factor: int = 50):
        self.g = g
        self.s = source
        n = g.n
        self.db: List[float] = [INF] * n
        self.db[self.s] = 0.0
        self.pred: List[Optional[int]] = [None] * n
        self.complete: Set[int] = set()  # mark when proven below current boundary

        # problem size and budgets
        self.m = sum(len(L) for L in g.adj)
        self.max_depth = max(1, int(max_depth))
        self.iter_cap_multiplier = max(1, int(iter_cap_multiplier))
        self.relax_cap = max(1000, int(relax_cap_factor * max(1, self.m)))
        self.relaxations = 0
        self.budget_exhausted = False

        # parameters (polylog in n)
        logn = max(1.0, math.log(max(2, n)))
        self.k = max(1, int(logn ** (1/3)))
        self.t = max(1, int(logn ** (2/3)))

    # ---------------- utilities ----------------
    def _note_relax(self):
        self.relaxations += 1
        if self.relaxations > self.relax_cap:
            self.budget_exhausted = True

    def relax(self, u: int, v: int, w: float, bound: float) -> bool:
        """Strict relaxation bounded by B, with budget accounting."""
        if self.budget_exhausted:
            return False
        nu = self.db[u] + w
        if nu < self.db[v] and nu < bound:
            self.db[v] = nu
            self.pred[v] = u
            self._note_relax()
            return True
        return False

    # ---------------- base case (Algorithm 2) ----------------
    def base_case(self, B: float, S: Set[int]) -> Tuple[float, Set[int]]:
        assert len(S) == 1
        (x,) = tuple(S)
        U0: Set[int] = set()
        H: List[Tuple[float, int]] = []
        if self.db[x] < B:
            heapq.heappush(H, (self.db[x], x))
        seen: Set[int] = set()
        while H and len(U0) < self.k + 1 and not self.budget_exhausted:
            du, u = heapq.heappop(H)
            if du != self.db[u]:
                continue
            if u in self.complete:
                continue
            U0.add(u)
            for v, w in self.g.adj[u]:
                if self.relax(u, v, w, B):
                    heapq.heappush(H, (self.db[v], v))
                    seen.add(v)
        if len(U0) <= self.k:
            return B, U0
        Bp = max(self.db[v] for v in U0)
        U = {v for v in U0 if self.db[v] < Bp}
        return Bp, U

    # ---------------- FindPivots (Algorithm 1) ----------------
    def find_pivots(self, B: float, S: Set[int]) -> Tuple[Set[int], Set[int]]:
        W = set(S)
        W_prev = set(S)
        for _ in range(self.k):
            if self.budget_exhausted:
                break
            Wi: Set[int] = set()
            for u in list(W_prev):
                for v, w in self.g.adj[u]:
                    # update upper bound, but only strict
                    if self.db[u] + w < self.db[v]:
                        self.db[v] = self.db[u] + w
                        self.pred[v] = u
                        self._note_relax()
                    if self.db[u] + w < B:
                        Wi.add(v)
            W |= Wi
            W_prev = Wi
        if len(W) > self.k * max(1, len(S)):
            return set(S), W
        # Build tight-edge forest on W
        children: Dict[int, List[int]] = {u: [] for u in W}
        indeg: Dict[int, int] = {u: 0 for u in W}
        for u in W:
            for v, w in self.g.adj[u]:
                if v in W and abs(self.db[v] - (self.db[u] + w)) <= 0.0:
                    children[u].append(v)
                    indeg[v] += 1
        roots = [u for u in W if indeg.get(u, 0) == 0]
        size: Dict[int, int] = {}
        def dfs_sz(x: int) -> int:
            s = 1
            for y in children.get(x, []):
                s += dfs_sz(y)
            size[x] = s
            return s
        for r in roots:
            dfs_sz(r)
        P = {u for u in S if size.get(u, 0) >= self.k}
        return P, W

    # ---------------- BMSSP (Algorithm 3) ----------------
    def bmssp(self, l: int, B: float, S: Set[int]) -> Tuple[float, Set[int]]:
        if self.budget_exhausted:
            return B, set()
        if l == 0:
            return self.base_case(B, S)
        P, W = self.find_pivots(B, S)
        if not P:
            # No pivots to expand; nothing to do at this level
            return B, {x for x in W if self.db[x] < B}
        # batch size; keep modest to avoid long pulls
        M = max(16, min(1024, 2 ** max(0, (l - 1) * self.t)))
        D = PriorityBatch(M)
        for x in P:
            if x not in self.complete and self.db[x] < B:
                D.insert(self.db[x], x)
        B0p = min((self.db[x] for x in P), default=B)
        U: Set[int] = set()
        iter_cap = self.iter_cap_multiplier * self.g.n
        no_progress_iters = 0
        prev_U_size = 0
        while len(U) < self.k * (2 ** (l * self.t)) and len(D) > 0:
            if self.budget_exhausted:
                break
            if iter_cap <= 0:
                break
            iter_cap -= 1
            Si, Bi = D.pull()
            if not Si:
                break
            Bip, Ui = self.bmssp(l - 1, Bi, set(Si))
            before = len(U)
            U |= Ui
            if len(U) == before:
                no_progress_iters += 1
            else:
                no_progress_iters = 0
            if no_progress_iters >= 3:
                break
            # relax outgoing and feed back
            K: List[Tuple[float, int]] = []
            for u in Ui:
                if u in self.complete:
                    continue
                for v, w in self.g.adj[u]:
                    newval = self.db[u] + w
                    if newval < self.db[v]:
                        self.db[v] = newval
                        self.pred[v] = u
                        self._note_relax()
                    if Bip <= newval < B and v not in self.complete:
                        if newval >= Bi:
                            D.insert(newval, v)
                        else:
                            K.append((newval, v))
            for x in Si:
                dx = self.db[x]
                if Bip <= dx < Bi and x not in self.complete:
                    K.append((dx, x))
            if K:
                D.batch_prepend(K)
            prev_U_size = len(U)
        # compute new boundary and mark completes
        Bp = min([B0p, B] + [self.db[x] for x in W if self.db[x] < B])
        newly_complete = {x for x in W if self.db[x] < Bp}
        self.complete.update(newly_complete)
        U |= newly_complete
        return Bp, U

    # ---------------- driver ----------------
    def run(self, allow_fallback: bool = True) -> Tuple[List[float], List[Optional[int]]]:
        n = self.g.n
        top_l = min(self.max_depth, max(1, math.ceil(math.log(max(2, n)) / max(1, self.t))))
        _B, _U = self.bmssp(top_l, INF, {self.s})
        # If some vertices remain at INF (disconnected) that's fine; otherwise return current db/pred
        return self.db, self.pred

# Public API

def sssp_break_sorting(g: Graph, source: int = 0, allow_fallback: bool = True) -> Tuple[List[float], List[Optional[int]]]:
    alg = SSSPBreaker(g, source)
    dist, pred = alg.run(allow_fallback=allow_fallback)
    return dist, pred

# ------------------------------
# Simple sanity test (will run if executed directly)
# ------------------------------
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
