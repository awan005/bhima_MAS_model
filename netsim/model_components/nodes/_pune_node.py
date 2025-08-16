from python_packages.pynsim.components import Node


class PuneNode(Node):
    """The Pune node class
    parent class that provides common methods for nodes in the Pune model.
    **Properties**:

        |  *institution_names* (list) - list of institutions associated with node

    """
    description = "Common Methods for Pune Nodes"

    def __init__(self, name, **kwargs):
        super(PuneNode, self).__init__(name, **kwargs)
        self.institution_names = []
        self.node_type = []

    def add_to_institutions(self, institution_list, n):
        """Add node to institutions.

        **Arguments**:

        |  *institution_list* (list) - list of institutions associated with node
        |  *n* (pynsim network) - network

        """
        for institution in n.institutions:
            for inst_name in institution_list:
                if institution.name.lower() == inst_name.lower():
                    institution.add_node(self)

    def get_depth(self, count=1):
        """calculates how far away this node is from the most upstream node."""

        if len(self.upstream_nodes) > 0:
            level = max([n.get_depth(count=count) for n in self.upstream_nodes])
        else:
            level = 0

        return count + level
