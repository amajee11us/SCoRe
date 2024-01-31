from .combinatorial.FL import FacilityLocation
from .combinatorial.GC import GraphCut
from .combinatorial.logdet import LogDet
from .combinatorial.snn_var import SubmodSNN
from .combinatorial.supcon_var import SubmodSupCon
from .combinatorial.triplet_var import SubmodTriplet

from .contrastive.supcon import SupConLoss
from .contrastive.triplet import TripletLoss
from .contrastive.liftedstructure import LiftedStructureLoss
from .contrastive.opl import OrthogonalProjectionLoss
from .contrastive.snn import SNNLoss
from .contrastive.npairs import NPairsLoss
from .contrastive.multiSimilarity import MSLoss

# PaCo losses, has to be used with the MoCo encoder
from .combinatorial.PaCoFL import PaCoFLLoss
from .combinatorial.PaCoGC import PaCoGCLoss

from .utils import *