"""
generate_diagram.py
===================
Generates the CDA Case 2 clustering pipeline architecture diagram.
Outputs: pipeline.dot, pipeline.svg, pipeline.pdf
"""
from pathlib import Path
import graphviz

OUT_DIR = Path(__file__).parent
OUT_NAME = "pipeline"

NODE_DEFAULTS = dict(
    shape="box",
    style="filled",
    fontname="Helvetica",
    fontsize="12",
)

C_NEUTRAL    = "#e8e8e8"   # input / output
C_PREPROCESS = "#c8e6c9"   # preprocessing
C_REDUCTION  = "#d1c4e9"   # dimensionality reduction
C_CLUSTER    = "#ffcdd2"   # clustering algorithms
C_KSELECT    = "#fff9c4"   # k-selection
C_EVAL       = "#b2ebf2"   # external evaluation
C_DISC       = "#ffe0b2"   # feature discriminability


def build_graph() -> graphviz.Digraph:
    g = graphviz.Digraph("cda2_pipeline")
    g.attr(rankdir="LR", splines="polyline", nodesep="0.5", ranksep="0.7",
           fontname="Helvetica")
    g.attr("node", **NODE_DEFAULTS)

    # ── Entry ──────────────────────────────────────────────────────
    g.node("input",
           label='<<B>HR_data_2.csv</B><BR/>'
                 '<FONT POINT-SIZE="10">312 obs (26 participants &#215; 4 rounds &#215; 3 phases)</FONT><BR/>'
                 '<FONT POINT-SIZE="10">71 columns (51 biosignal + 20 metadata/questionnaire)</FONT>>',
           fillcolor=C_NEUTRAL)

    # ── Preprocessing ──────────────────────────────────────────────
    with g.subgraph(name="cluster_preprocess") as pre:
        pre.attr(label="Preprocessing", style="dashed", color="#aaaaaa",
                 fontname="Helvetica", fontsize="11", labelloc="t", labeljust="l")

        pre.node("exclude",
                 label='<<B>Exclude</B><BR/>'
                       '<FONT POINT-SIZE="10">metadata + questionnaire items</FONT>>',
                 fillcolor=C_PREPROCESS)
        pre.node("impute",
                 label='<<B>Imputation</B><BR/>'
                       '<FONT POINT-SIZE="10">within-Phase group mean</FONT>>',
                 fillcolor=C_PREPROCESS)
        pre.node("dedup",
                 label='<<B>Redundancy Removal</B><BR/>'
                       '<FONT POINT-SIZE="10">drop |r| &gt; 0.95</FONT>>',
                 fillcolor=C_PREPROCESS)
        pre.node("norm",
                 label='<<B>Individual Normalisation</B><BR/>'
                       '<FONT POINT-SIZE="10">z-score per participant</FONT><BR/>'
                       '<FONT POINT-SIZE="10"><I>states, not traits</I></FONT>>',
                 fillcolor=C_PREPROCESS)

        pre.edge("exclude", "impute")
        pre.edge("impute",  "dedup")
        pre.edge("dedup",   "norm")

    # ── Dimensionality Reduction ───────────────────────────────────
    with g.subgraph(name="cluster_reduction") as red:
        red.attr(label="Dimensionality Reduction", style="dashed", color="#aaaaaa",
                 fontname="Helvetica", fontsize="11", labelloc="t", labeljust="l")

        with red.subgraph() as r:
            r.attr(rank="same")
            r.node("pca",
                   label='<<B>PCA</B><BR/>'
                         '<FONT POINT-SIZE="10">16 components</FONT><BR/>'
                         '<FONT POINT-SIZE="10">&#8805;80% variance</FONT>>',
                   fillcolor=C_REDUCTION)
            r.node("spca",
                   label='<<B>SparsePCA</B><BR/>'
                         '<FONT POINT-SIZE="10">16 components, &#945;=1</FONT><BR/>'
                         '<FONT POINT-SIZE="10">L1-sparse loadings</FONT>>',
                   fillcolor=C_REDUCTION)
            r.node("umap",
                   label='<<B>UMAP</B><BR/>'
                         '<FONT POINT-SIZE="10">10 components</FONT><BR/>'
                         '<FONT POINT-SIZE="10">n_neighbors=50, min_dist=0</FONT>>',
                   fillcolor=C_REDUCTION)

    # Hub node — reduction → clustering (avoids 3×5 = 15 arrows)
    g.node("hub_red",
           label="", shape="point", width="0.15",
           style="filled", fillcolor="#888888")

    # ── Clustering ─────────────────────────────────────────────────
    with g.subgraph(name="cluster_clustering") as cl:
        cl.attr(label="Clustering", style="dashed", color="#aaaaaa",
                fontname="Helvetica", fontsize="11", labelloc="t", labeljust="l")

        with cl.subgraph() as r:
            r.attr(rank="same")
            r.node("hier",
                   label='<<B>Hierarchical</B><BR/>'
                         '<FONT POINT-SIZE="10">Ward linkage</FONT>>',
                   fillcolor=C_CLUSTER)
            r.node("kmeans",
                   label='<<B>K-Means</B>>',
                   fillcolor=C_CLUSTER)
            r.node("kmed",
                   label='<<B>K-Medoids</B><BR/>'
                         '<FONT POINT-SIZE="10">PAM</FONT>>',
                   fillcolor=C_CLUSTER)
            r.node("gmm",
                   label='<<B>GMM</B><BR/>'
                         '<FONT POINT-SIZE="10">full / tied / diag / sph</FONT>>',
                   fillcolor=C_CLUSTER)
            r.node("dbscan",
                   label='<<B>DBSCAN</B><BR/>'
                         '<FONT POINT-SIZE="10">grid search &#949;, min_samples</FONT>>',
                   fillcolor=C_CLUSTER)

        # Force vertical order (reversed chain for LR rankdir): dbscan at bottom
        cl.edge("dbscan", "gmm",    style="invis")
        cl.edge("gmm",    "kmed",   style="invis")
        cl.edge("kmed",   "kmeans", style="invis")
        cl.edge("kmeans", "hier",   style="invis")

    # ── k-Selection (two separate nodes) ──────────────────────────
    with g.subgraph() as ks:
        ks.attr(rank="same")
        ks.node("kselect_sil",
                label='<<B>k-Selection</B><BR/>'
                      '<FONT POINT-SIZE="10">Silhouette score</FONT><BR/>'
                      '<FONT POINT-SIZE="10">Hierarchical, K-Means, K-Medoids</FONT>>',
                fillcolor=C_KSELECT)
        ks.node("kselect_bic",
                label='<<B>k-Selection</B><BR/>'
                      '<FONT POINT-SIZE="10">BIC</FONT><BR/>'
                      '<FONT POINT-SIZE="10">GMM</FONT>>',
                fillcolor=C_KSELECT)

    # ── Cluster Assignments ────────────────────────────────────────
    g.node("assignments",
           label='<<B>Cluster Assignments</B><BR/>'
                 '<FONT POINT-SIZE="10">hard labels (K-Means, K-Medoids, Hier, DBSCAN)</FONT><BR/>'
                 '<FONT POINT-SIZE="10">soft probabilities (GMM)</FONT>>',
           fillcolor=C_NEUTRAL)

    # ── External Evaluation ────────────────────────────────────────
    g.node("ext_eval",
           label='<<B>External Evaluation</B><BR/>'
                 '<FONT POINT-SIZE="10">ARI</FONT><BR/>'
                 '<FONT POINT-SIZE="10">vs Phase / Round / Cohort / Role</FONT>>',
           fillcolor=C_EVAL)

    # ── Feature Discriminability ───────────────────────────────────
    g.node("disc",
           label='<<B>Feature Discriminability</B><BR/>'
                 '<FONT POINT-SIZE="10">Mann-Whitney U + Cohen\'s d</FONT><BR/>'
                 '<FONT POINT-SIZE="10">on original normalised features</FONT>>',
           fillcolor=C_DISC)

    # ── Legend ─────────────────────────────────────────────────────
    with g.subgraph(name="cluster_legend") as leg:
        leg.attr(label="", style="dashed", color="#aaaaaa")
        leg.node("leg_pre",  label="Preprocessing",    fillcolor=C_PREPROCESS,
                 fontsize="9", width="1.3", height="0.2", margin="0.04,0.01")
        leg.node("leg_red",  label="Dim. Reduction",   fillcolor=C_REDUCTION,
                 fontsize="9", width="1.3", height="0.2", margin="0.04,0.01")
        leg.node("leg_cl",   label="Clustering",       fillcolor=C_CLUSTER,
                 fontsize="9", width="1.3", height="0.2", margin="0.04,0.01")
        leg.node("leg_k",    label="k-Selection",      fillcolor=C_KSELECT,
                 fontsize="9", width="1.3", height="0.2", margin="0.04,0.01")
        leg.node("leg_eval", label="Evaluation",       fillcolor=C_EVAL,
                 fontsize="9", width="1.3", height="0.2", margin="0.04,0.01")
        leg.node("leg_disc", label="Discriminability", fillcolor=C_DISC,
                 fontsize="9", width="1.3", height="0.2", margin="0.04,0.01")
        leg.edge("leg_pre",  "leg_red",  style="invis")
        leg.edge("leg_red",  "leg_cl",   style="invis")
        leg.edge("leg_cl",   "leg_k",    style="invis")
        leg.edge("leg_k",    "leg_eval", style="invis")
        leg.edge("leg_eval", "leg_disc", style="invis")

    # ── Edges ──────────────────────────────────────────────────────
    # Input → preprocessing chain
    g.edge("input", "exclude")

    # Preprocessing → all three reductions
    g.edge("norm", "pca")
    g.edge("norm", "spca")
    g.edge("norm", "umap")

    # Reductions → hub → clustering (3+5 edges instead of 15)
    for rn in ["pca", "spca", "umap"]:
        g.edge(rn, "hub_red", arrowhead="none")
    for cn in ["hier", "kmeans", "kmed", "gmm", "dbscan"]:
        g.edge("hub_red", cn)

    # Clustering → k-selection (Silhouette for centroid/hierarchical, BIC for GMM)
    for cn in ["hier", "kmeans", "kmed"]:
        g.edge(cn, "kselect_sil")
    g.edge("gmm", "kselect_bic")

    # DBSCAN bypasses k-selection → directly to assignments
    g.edge("dbscan", "assignments", constraint="false")

    g.edge("kselect_sil", "assignments")
    g.edge("kselect_bic", "assignments")
    g.edge("assignments", "ext_eval")
    g.edge("assignments", "disc")

    return g


def main():
    g = build_graph()

    dot_path = OUT_DIR / OUT_NAME
    g.save(str(dot_path) + ".dot")
    print(f"DOT  → {dot_path}.dot")

    g.render(filename=str(dot_path), format="svg", cleanup=False)
    print(f"SVG  → {dot_path}.svg")

    g.render(filename=str(dot_path), format="pdf", cleanup=False)
    print(f"PDF  → {dot_path}.pdf")


if __name__ == "__main__":
    main()
