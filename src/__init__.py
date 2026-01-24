"""
SMCMTI package: core utilities for moment tensor inversion.

This package exposes:

- Forward modelling (NumPy):
    - station_angles
    - forward_amplitude
- Forward modelling (PyTensor):
    - pt_station_angles
    - pt_forward_amplitude
- Data preparation:
    - polarity_matrix
    - amplitude_ratio_matrix
- Likelihoods:
    - polarity_ln_pdf
    - amplitude_ratio_ln_pdf
- Tape & Tape parameterisation (NumPy):
    - Tape_MT6
    - Tape_MT33
    - MT33_MT6
    - GD_E
- Tape & Tape parameterisation (PyTensor):
    - pt_Tape_MT6
    - pt_Tape_MT33
    - pt_MT33_MT6
    - pt_GD_E
- Inversion classes:
    - Inversion (NumPy-based, uses custom Op)
    - InversionPyTensor (PyTensor-based, compilable, parallelizable)
"""

from .forward_model import station_angles, forward_amplitude
from .forward_model_pytensor import pt_station_angles, pt_forward_amplitude
from .data_prep import polarity_matrix, amplitude_ratio_matrix
from .likelihoods import polarity_ln_pdf, amplitude_ratio_ln_pdf
from .tape import Tape_MT6, Tape_MT33, MT33_MT6, GD_E
from .tape_pytensor import pt_Tape_MT6, pt_Tape_MT33, pt_MT33_MT6, pt_GD_E
from .inversion_pytensor import InversionPyTensor

__all__ = [
    # NumPy versions
    "station_angles",
    "forward_amplitude",
    "Tape_MT6",
    "Tape_MT33",
    "MT33_MT6",
    "GD_E",
    # PyTensor versions
    "pt_station_angles",
    "pt_forward_amplitude",
    "pt_Tape_MT6",
    "pt_Tape_MT33",
    "pt_MT33_MT6",
    "pt_GD_E",
    # Data and likelihoods
    "polarity_matrix",
    "amplitude_ratio_matrix",
    "polarity_ln_pdf",
    "amplitude_ratio_ln_pdf",
    # Inversion classes
    "InversionPyTensor",
]
