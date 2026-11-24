"""Inference helpers from cold_leaf_check.py."""

from terminal_tree_controller import TerminalPolicy, TerminalSearch
class ColdSearch(TerminalSearch):
    leaf_mode='trained'
    def configure_request(self,public,limits):
        result=super().configure_request(public,limits)
        if self.leaf_mode!='control':result['leaf_actor']=self.leaf_mode
        return result


class ColdPolicy(TerminalPolicy):
    search_class=ColdSearch
