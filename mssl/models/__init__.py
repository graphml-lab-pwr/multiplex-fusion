from .custom import F_DGI, F_GBT
from .deepwalk import DeepWalk, DeepWalkDataModule, FlattenedGraphDeepWalk
from .dgi import DGI, FlattenedGraphDGI
from .dmgi import DMGI
from .hdgi import HDGI
from .mhgcn import MHGCN
from .s2mgrl import S2MGRL
from .supervised_gnn import (
    FlattenedGraphSupervisedGAT,
    FlattenedGraphSupervisedGCN,
    SupervisedGAT,
    SupervisedGCN,
)

__all__ = [
    "DeepWalk",
    "DeepWalkDataModule",
    "DGI",
    "DMGI",
    "FlattenedGraphDeepWalk",
    "FlattenedGraphDGI",
    "FlattenedGraphSupervisedGAT",
    "FlattenedGraphSupervisedGCN",
    "F_DGI",
    "F_GBT",
    "HDGI",
    "MHGCN",
    "S2MGRL",
    "SupervisedGAT",
    "SupervisedGCN",
]
