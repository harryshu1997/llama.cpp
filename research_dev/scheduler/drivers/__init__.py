"""Device drivers of the portable path: one interface per device class.

A site (``configuration/site.py``) names a host and its phones; the drivers here turn that description
into the few operations a run needs -- launch a server on a GPU by index, start and stop a phone helper,
read health, meter energy, gate a run on a quiet host -- so that a new host or a new phone is a new site
file, not a fork. Import rule: ``configuration``, ``_internal`` and ``adapters`` sit below this package;
nothing here imports ``campaigns``, ``_unified`` or ``scheduler.py``.
"""
