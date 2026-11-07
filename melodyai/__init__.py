"""MelodyAI: query-by-humming, whistling and lyrics song recognition.

The package is organised as independent, swappable layers:

- ``melodyai.audio``      decoding, validation, preprocessing, quality estimation, synthesis
- ``melodyai.pitch``      f0 extractors (pYIN, Basic Pitch, torchcrepe) and note segmentation
- ``melodyai.features``   melody representations (contours, windows, chroma, intervals)
- ``melodyai.matching``   DTW kernels and melody matching algorithms
- ``melodyai.lyrics``     ASR wrapper, text normalisation and fuzzy lyrics search
- ``melodyai.fusion``     score fusion and abstention
- ``melodyai.indexing``   catalog feature store and incremental indexes
- ``melodyai.models``     neural melody embedding model
- ``melodyai.training``   contrastive training pipeline
- ``melodyai.evaluation`` reproducible benchmarks and metrics
- ``melodyai.demo``       original synthetic demo catalog generator
"""

__version__ = "0.1.0"
