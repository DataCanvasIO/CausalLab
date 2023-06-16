from itertools import product

import numpy as np
import pandas as pd

from ylearn.bayesian import _base
from ylearn.bayesian._dag import DiGraph


def _alg_notears(data):
    from ylearn.causal_discovery._discovery import CausalDiscovery
    cd = CausalDiscovery(hidden_layer_dim=[data.shape[1], ])
    return cd(data)


def _alg_pc_stable(data):
    from ylearn.causal_discovery._proxy_gcastle import GCastleProxy
    cd = GCastleProxy(learner='PC', variant='stable')
    return cd(data)


def _alg_pc_original(data):
    from ylearn.causal_discovery._proxy_gcastle import GCastleProxy
    cd = GCastleProxy(learner='PC', variant='original')
    return cd(data)


def _alg_ges_bdeu(data):
    from ylearn.causal_discovery._proxy_gcastle import GCastleProxy
    cd = GCastleProxy(learner='GES', criterion='bdeu')
    return cd(data)


def _alg_ges_bic(data):
    from ylearn.causal_discovery._proxy_gcastle import GCastleProxy
    cd = GCastleProxy(learner='GES', criterion='bic')
    return cd(data)


def _alg_icalingam(data):
    from ylearn.causal_discovery._proxy_gcastle import GCastleProxy
    cd = GCastleProxy(learner='ICALiNGAM')
    return cd(data)


def _alg_mcsl(data):
    from ylearn.causal_discovery._proxy_gcastle import GCastleProxy
    cd = GCastleProxy(learner='MCSL')
    return cd(data)


def _alg_grandag(data):
    from ylearn.causal_discovery._proxy_gcastle import GCastleProxy
    cd = GCastleProxy(learner='GraNDAG', input_dim=data.shape[1])
    return cd(data)


discoverers = {
    'PC(Stable)': _alg_pc_stable,
    'PC(Original)': _alg_pc_original,
    'GES(bedu)': _alg_ges_bdeu,
    'GES(bid)': _alg_ges_bic,
    'ICALiNGAM': _alg_icalingam,
    # 'MCSL': _alg_mcsl,
    # 'GraNDAG': _alg_grandag,
    'NoTears': _alg_notears,
}


class CausationHolder:
    """
    Parameters
    ----------
    node_states: dict, key is node name, value is node state

    Attributes
    ----------
    position: dict, key is node name, value is node position tuple (x,y)
    threshold: int
    matrices: matrix found by discovery algorithms
    enabled: list of tuple(cause,effect)
    disabled: list of tuple(cause,effect)
    """

    def __init__(self, node_states):
        assert isinstance(node_states, dict)

        self.node_states = node_states
        self.position = {}
        self.threshold = 1
        self.matrices = {}
        self.enabled = []
        self.disabled = []

    def reset(self):
        self.threshold = 1
        self.matrices = {}
        self.enabled = set()
        self.disabled = set()

    def add_matrix(self, name, matrix):
        assert isinstance(name, str)
        assert isinstance(matrix, pd.DataFrame)
        assert set(matrix.columns.tolist()) == set(matrix.index.tolist())
        assert set(matrix.columns.tolist()).issubset(set(self.node_states.keys()))

        self.matrices[name] = matrix

    def disable(self, cause, effect):
        assert cause in set(self.node_states.keys())
        assert effect in set(self.node_states.keys())

        if (cause, effect) in self.enabled:
            self.enabled.remove((cause, effect))
        self.disabled.append((cause, effect))

    def enable(self, cause, effect):
        assert cause in set(self.node_states.keys())
        assert effect in set(self.node_states.keys())

        if (cause, effect) in self.disabled:
            self.disabled.remove((cause, effect))
        self.enabled.append((cause, effect))

    def remove_disabled(self, cause, effect):
        assert cause in set(self.node_states.keys())
        assert effect in set(self.node_states.keys())

        if (cause, effect) in self.disabled:
            self.disabled.remove((cause, effect))

    def remove_enabled(self, cause, effect):
        assert cause in set(self.node_states.keys())
        assert effect in set(self.node_states.keys())

        if (cause, effect) in self.enabled:
            self.enabled.remove((cause, effect))

    @property
    def is_empty(self):
        """
        Weather cause-effect does not exist.
        """
        return len(self.matrices) == 0 and len(self.enabled) == 0

    @property
    def causal_matrix(self):
        """
        get the final causal matrix
        """
        matrix = None
        if len(self.matrices) > 0:
            for m in self.matrices.values():
                if matrix is None:
                    matrix = m
                else:
                    matrix = matrix + m
            if self.threshold is not None and self.threshold > 0:
                values = np.where(matrix.values >= self.threshold, matrix.values, 0)
                matrix = pd.DataFrame(values, columns=matrix.columns, index=matrix.index)
            for c, e in self.disabled:
                matrix[e][c] = 0
        else:
            nodes = self.node_states.keys()
            matrix = pd.DataFrame(np.zeros((len(nodes), len(nodes)), dtype='int'),
                                  columns=nodes, index=nodes)

        for c, e in self.enabled:
            if matrix[e][c] < 1:
                matrix[e][c] = 1

        return matrix.copy()

    @property
    def graph(self):
        """
        return the DiGraph object.

        edge attributes:
        -----------------
        weight: int, the number of algorithms found the relation
        expert: int, 1: enabled by expert, 0: enabled by discovery algorithms
        """

        def node_attribute(n):
            shape = 'box' if isinstance(self.node_states[n], _base.CategoryNodeState) else 'ellipse'
            attr = dict(shape=shape)

            if self.position is not None and n in self.position.keys():
                x, y = self.position[n]
                attr['x'] = x
                attr['y'] = y
            return attr

        m = self.causal_matrix
        columns = m.columns.tolist()
        nodes = [(n, node_attribute(n)) for n in columns]
        edges = [(c, e,
                  dict(weight=m[e][c],
                       expert=int((c, e) in self.enabled),
                       )
                  )
                 for c, e in product(columns, columns)
                 if c != e and m[e][c] > 0
                 ]  # list of tuple(start, end, edge_data)

        g = DiGraph(None)
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)

        return g
