import torch
import numpy as np
from pcst_fast import pcst_fast
from torch_geometric.data.data import Data

def retrieval_via_pcst(graph, q_emb, textual_nodes, textual_edges, topk=3, topk_e=3, cost_e=0.5):
    def _to_edges_csv(df):
        wanted = ['src', 'edge_attr', 'dst']
        if all(c in df.columns for c in wanted):
            return df.to_csv(index=False, columns=wanted)
        return df.to_csv(index=False)

    nodes_csv = textual_nodes.to_csv(index=False) if len(textual_nodes) > 0 else ""
    edges_csv = _to_edges_csv(textual_edges) if len(textual_edges) > 0 else ""
    desc = nodes_csv + ("\n" if nodes_csv and edges_csv else "") + edges_csv

    data = Data(
        x=graph.x,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        num_nodes=graph.num_nodes
    )
    return data, desc