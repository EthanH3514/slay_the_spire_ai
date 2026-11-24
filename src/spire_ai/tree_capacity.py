"""Storage capacity for the existing bounded, network-only tree search."""
from dataclasses import dataclass,asdict,replace

VERSION='simulation-derived-tree-capacity-v1'
MAX_NEW_NODES=3
ARENA_BYTES=64*1024*1024


@dataclass(frozen=True)
class TreeCapacity:
    simulations:int
    max_nodes:int
    arena_bytes:int=ARENA_BYTES

    @classmethod
    def for_simulations(cls,count):
        if type(count) is not int or not 1<=count<=4096:
            raise ValueError('invalid tree simulation allowance')
        return cls(count,1+MAX_NEW_NODES*count)

    def config(self,previous):
        if previous.training_rollouts!=self.simulations:
            raise ValueError('capacity must match simulation allowance')
        return replace(previous,max_nodes=self.max_nodes,arena_bytes=self.arena_bytes)

    def wire(self):
        return dict(version=VERSION,**asdict(self),max_new_nodes_per_simulation=MAX_NEW_NODES)

    def validate(self,native):
        if native.get('forced'):return
        t=native['tree']
        if (t['arena_bytes']!=self.arena_bytes or t['simulation_cap']!=self.simulations
                or t['new_node_cap']!=MAX_NEW_NODES
                or not 0<t['nodes']<=self.max_nodes
                or not 0<t['retained_payload_bytes']<=self.arena_bytes
                or t['max_new_nodes_per_simulation']>MAX_NEW_NODES):
            raise ValueError('tree capacity accounting mismatch')
        if t['expansion_closed']:
            raise ValueError('tree capacity exhausted before completing declared search; no silent shallow result')
