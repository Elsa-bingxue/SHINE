"""SHINE spatial multi-omics integration."""

__version__ = '0.1.0'
__all__ = ['SHINE', 'prepare', 'PreparedData', 'RevisionSHINE',
           'normalize_counts', 'build_graphs', 'prepare_paired']


def __getattr__(name):
    # Importing package metadata must not parse arguments or start training.
    if name in ('SHINE', 'prepare', 'PreparedData'):
        from . import api
        return getattr(api, name)
    if name == 'RevisionSHINE':
        from .revision import RevisionSHINE
        return RevisionSHINE
    if name in ('normalize_counts', 'build_graphs', 'prepare_paired'):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError('module shine has no attribute ' + repr(name))
