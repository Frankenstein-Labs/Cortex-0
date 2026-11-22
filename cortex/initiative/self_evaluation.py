"""Self-evaluation over recorded reasoning runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..memory.store import MemoryStore


@dataclass
class EvaluationReport:
    total: int
    verified: int
    accuracy: float
    failures: List[dict]
    recommendations: List[str]

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "verified": self.verified,
            "accuracy": round(self.accuracy, 4),
            "failures": self.failures[:5],
            "recommendations": self.recommendations,
        }


class SelfEvaluator:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def report(self, limit: int = 50) -> EvaluationReport:
        runs = self.store.reasoning_runs(limit=limit)
        total = len(runs)
        verified = sum(1 for run in runs if run.get("verified"))
        accuracy = verified / total if total else 0.0
        failures = [run for run in runs if not run.get("verified")][:5]
        recommendations: List[str] = []
        if total == 0:
            recommendations.append("Aucune exécution de raisonnement enregistrée : lancer une évaluation.")
        elif accuracy < 0.4:
            recommendations.append("Accuracy faible : augmenter les données arithmétiques et vérifier le format de réponse.")
        elif accuracy < 0.7:
            recommendations.append("Accuracy intermédiaire : ajouter des exemples de vérification et de révision.")
        else:
            recommendations.append("Accuracy correcte sur ce petit suite : élargir les types de problèmes.")
        if failures:
            recommendations.append("Analyser les échecs récents et ajouter des traces corrigées au jeu SFT.")
        return EvaluationReport(total, verified, accuracy, failures, recommendations)

    def initiative(self, now: Optional[float] = None) -> Optional[Dict[str, object]]:
        report = self.report()
        if report.total < 5:
            return None
        return {
            "kind": "self_evaluation",
            "message": (
                f"Auto-évaluation : {report.accuracy:.0%} de réponses validées sur "
                f"{report.total} raisonnements. {report.recommendations[0]}"
            ),
            "importance": 0.65 if report.accuracy < 0.7 else 0.45,
            "source": "reasoning_runs",
        }
