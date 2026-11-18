"""The portable run path: one host, one llama-server, optional helper phones, on any site.

``discover`` and ``doctor`` describe and check a site; ``onboard-host`` and ``onboard-phone`` measure it
and write a site profile; ``provision`` puts the planned shards on a phone; ``run`` replays a schedule
alone or with shared layers; ``report`` turns a run directory into tables. Everything device-specific
comes from the site file and the drivers (``research_dev/scheduler/drivers``).
"""
