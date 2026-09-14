"""Batch append physical collider IDs, preserving existing prefixes and padding."""
import numpy as np


def append_collider_paths(paths, destinations, starts, lengths, payload, atom_offset=0):
    # Most outgoing rays have no new collision. Build indices only for the
    # nonempty physical paths, not every ray or every reserved cap slot.
    active=np.flatnonzero(lengths)
    if not len(active):
        return
    row,column=np.nonzero(np.arange(int(lengths[active].max()))[None,:] < lengths[active,None])
    source_rows=active[row]
    target_rows=destinations[source_rows]
    paths[target_rows,starts[target_rows]+column]=payload[source_rows,column]+atom_offset
