from main_generator import *
from arangodb import save_rule_database, save_topology
from topology_traffic_generator import *


def main():


















    node_list, nc_list, lightpath_list, node_DG, board_DG = process_traffic("outputs/static_topology/training topo/topology_15_nodes_40_lightpaths.txt")
    OTS_layer_list, OMS_layer_list, OCh_layer_list = process_trails(lightpath_list, node_DG, board_DG)
    save_topology(node_list, nc_list, board_DG, OMS_layer_list)



main()
