"""MEMENTO: Momentum-Memory Poisoning and Continuum-Momentum Defense for OTA-FL.

Package layout
--------------
momento.data        datasets, non-IID partitions, backdoor triggers
momento.models      functional models with flat parameter vectors
momento.channel     over-the-air (AirComp) uplink physics
momento.server      server optimizers (FedAvg/M/Adam/Yogi), OTA defenses, CMA
momento.attacks     backdoor attacks incl. the momentum-echo family
momento.client      local training
momento.simulator   the round loop
momento.metrics     MA, ASR, lifespan, detection statistics
momento.records     JSON run records and hashing
"""

__version__ = "0.1.0"
