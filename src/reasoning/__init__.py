"""
Reasoning Subsystem for IVARA.

Exports:
- ContradictionAnalyzer, ContradictionType, ContradictionResult
- EvidenceIntegrator, EvidenceRole, MemoryEvidenceProfile
- EvidenceGate, GatingDecision, EvidenceGateReport
"""

from .contradiction import ContradictionAnalyzer, ContradictionType, ContradictionResult
from .evidence_integrator import EvidenceIntegrator, EvidenceRole, MemoryEvidenceProfile
from .evidence_gate import EvidenceGate, GatingDecision, EvidenceGateReport

__all__ = [
    "ContradictionAnalyzer",
    "ContradictionType",
    "ContradictionResult",
    "EvidenceIntegrator",
    "EvidenceRole",
    "MemoryEvidenceProfile",
    "EvidenceGate",
    "GatingDecision",
    "EvidenceGateReport"
]
