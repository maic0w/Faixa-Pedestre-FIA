"""Modulo de Decisao e Calculo do Indice de Conflito Observacional (ICO).

Responsavel por agregar metricas substitutas de seguranca (SSMs) como PET critico,
tempo de espera normalizado e taxa de nao-cedencia em um score sintetico de 0 a 100
para apoio a tomada de decisao e priorizacao de vistorias tecnicas da SMTT.

Enquadramento Epistemico:
O ICO e uma medida observacional de criticidade relativa para triagem e direcionamento
de equipes de fiscalizacao e engenharia viaria. Nao representa probabilidade estatistica
ou risco absoluto de acidente/colisao.
"""

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("PedestrianConflict.Decision")


@dataclass
class ConflictEvent:
    """Representacao de um evento observacional de interacao entre pedestre e veiculo."""
    event_id: str
    pedestrian_track_id: int
    vehicle_track_id: Optional[int]
    roi_id: str
    t_entry_pedestrian: float
    t_exit_pedestrian: float
    t_entry_vehicle: Optional[float]
    pedestrian_wait_time_s: float = 0.0
    vehicle_yielded: bool = True
    is_valid: bool = True
    pet_critical_threshold_s: float = 2.0

    @property
    def pet_s(self) -> Optional[float]:
        """Calcula o Post-Encroachment Time (PET = t_entry_veiculo - t_exit_pedestre)."""
        if self.t_entry_vehicle is None or self.t_exit_pedestrian is None:
            return None
        return round(self.t_entry_vehicle - self.t_exit_pedestrian, 3)

    @property
    def is_pet_critical(self) -> bool:
        """Verifica se o PET observado viola a margem operacional de proximidade (< 2.0s)."""
        pet = self.pet_s
        if pet is None:
            return False
        return pet < self.pet_critical_threshold_s


@dataclass
class ICOWeights:
    """Pesos do modelo linear ponderado do ICO."""
    w_pet: float = 0.50
    w_wait: float = 0.25
    w_yield: float = 0.25

    def __post_init__(self):
        total = self.w_pet + self.w_wait + self.w_yield
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(f"A soma dos pesos deve ser 1.0 (obtido: {total:.4f})")
        if any(w < 0 for w in (self.w_pet, self.w_wait, self.w_yield)):
            raise ValueError("Todos os pesos devem ser nao-negativos.")


class ICOCalculator:
    """Calculador do Indice de Conflito Observacional e classificador de criticidade para apoio a decisao."""

    def __init__(
        self,
        weights: Optional[ICOWeights] = None,
        t_max_wait_s: float = 30.0,
        pet_critical_threshold_s: float = 2.0,
    ):
        self.weights = weights or ICOWeights()
        self.t_max_wait_s = t_max_wait_s
        self.pet_critical_threshold_s = pet_critical_threshold_s

    @staticmethod
    def classify_risk(ico_score: float) -> str:
        """Classifica o score do ICO em faixas de criticidade estabelecidas no guia tecnico.

        - [0, 35): Baixo (Travessia fluida; condutores respeitam a prioridade; monitoramento de rotina)
        - [35, 65): Moderado (Retencao pontual ou desrespeito eventual; vistoria de sinalizacao/iluminacao)
        - [65, 100]: Critico (Conflitos frequentes de PET e alta retencao; prioridade de engenharia)
        """
        if ico_score < 35.0:
            return "Baixo"
        elif ico_score < 65.0:
            return "Moderado"
        else:
            return "Crítico"

    def calculate_from_rates(
        self,
        p_crit: float,
        e_norm: float,
        d_faixa: float,
        weights: Optional[ICOWeights] = None,
    ) -> float:
        """Calcula o escore sintético do ICO a partir das taxas observacionais primarias [0, 1]."""
        w = weights or self.weights
        p_crit_clamped = np.clip(p_crit, 0.0, 1.0)
        e_norm_clamped = np.clip(e_norm, 0.0, 1.0)
        d_faixa_clamped = np.clip(d_faixa, 0.0, 1.0)

        score = 100.0 * (
            w.w_pet * p_crit_clamped +
            w.w_wait * e_norm_clamped +
            w.w_yield * d_faixa_clamped
        )
        return float(np.round(score, 2))

    def aggregate_session(
        self,
        video_id: str,
        events: List[ConflictEvent],
    ) -> Dict[str, any]:
        """Processa uma lista de eventos e produz o sumario com metricas e ICO (Contrato C)."""
        valid_events = [e for e in events if e.is_valid]
        total_crossings = len(events)
        eligible_events = len(valid_events)

        if eligible_events == 0:
            logger.info("Sessao %s sem travessias elegiveis. ICO fixado em 0.0 (Baixo).", video_id)
            return {
                "video_id": video_id,
                "total_crossings": total_crossings,
                "eligible_events": 0,
                "valid_pet_count": 0,
                "critical_pet_count": 0,
                "p_crit": 0.0,
                "mean_wait_time_s": 0.0,
                "e_norm": 0.0,
                "non_yield_count": 0,
                "d_faixa": 0.0,
                "ico_score": 0.0,
                "risk_level": "Baixo",
                "epistemic_note": "Indice de Conflito Observacional (SAD) para priorizacao tecnica, sem inferencia probabilistica de sinistro.",
            }

        # 1. Proporção de eventos críticos de PET (P_crit)
        events_with_pet = [e for e in valid_events if e.pet_s is not None]
        valid_pet_count = len(events_with_pet)
        critical_pet_count = sum(1 for e in events_with_pet if e.is_pet_critical)
        p_crit = critical_pet_count / eligible_events if eligible_events > 0 else 0.0

        # 2. Tempo de espera médio normalizado (E_norm)
        wait_times = [e.pedestrian_wait_time_s for e in valid_events]
        mean_wait_time = float(np.mean(wait_times)) if wait_times else 0.0
        e_norm = min(1.0, mean_wait_time / self.t_max_wait_s)

        # 3. Taxa de não-cedência na faixa (D_faixa)
        non_yield_count = sum(1 for e in valid_events if not e.vehicle_yielded)
        d_faixa = non_yield_count / eligible_events if eligible_events > 0 else 0.0

        # Score do ICO e classificação em faixa de criticidade
        ico_score = self.calculate_from_rates(p_crit, e_norm, d_faixa)
        risk_level = self.classify_risk(ico_score)

        return {
            "video_id": video_id,
            "total_crossings": total_crossings,
            "eligible_events": eligible_events,
            "valid_pet_count": valid_pet_count,
            "critical_pet_count": critical_pet_count,
            "p_crit": round(p_crit, 4),
            "mean_wait_time_s": round(mean_wait_time, 2),
            "e_norm": round(e_norm, 4),
            "non_yield_count": non_yield_count,
            "d_faixa": round(d_faixa, 4),
            "ico_score": ico_score,
            "risk_level": risk_level,
            "epistemic_note": "Indice de Conflito Observacional (SAD) para priorizacao tecnica, sem inferencia probabilistica de sinistro.",
        }

    def sensitivity_analysis(
        self,
        p_crit: float,
        e_norm: float,
        d_faixa: float,
        variations: Tuple[float, ...] = (-0.20, -0.10, 0.10, 0.20),
    ) -> List[Dict[str, any]]:
        """Realiza analise de sensibilidade nos pesos (variacoes de +/-10% e +/-20%)."""
        base_score = self.calculate_from_rates(p_crit, e_norm, d_faixa)
        results = [{
            "variation": "0% (Base)",
            "w_pet": self.weights.w_pet,
            "w_wait": self.weights.w_wait,
            "w_yield": self.weights.w_yield,
            "ico_score": base_score,
            "risk_level": self.classify_risk(base_score),
        }]

        for delta in variations:
            new_w_pet = self.weights.w_pet * (1.0 + delta)
            remaining = 1.0 - new_w_pet
            sum_other = self.weights.w_wait + self.weights.w_yield
            if sum_other > 0:
                new_w_wait = remaining * (self.weights.w_wait / sum_other)
                new_w_yield = remaining * (self.weights.w_yield / sum_other)
            else:
                new_w_wait = remaining / 2
                new_w_yield = remaining / 2

            try:
                temp_weights = ICOWeights(
                    w_pet=new_w_pet,
                    w_wait=new_w_wait,
                    w_yield=new_w_yield,
                )
                score = self.calculate_from_rates(p_crit, e_norm, d_faixa, temp_weights)
                results.append({
                    "variation": f"{delta * 100:+.0f}% w_pet",
                    "w_pet": round(new_w_pet, 4),
                    "w_wait": round(new_w_wait, 4),
                    "w_yield": round(new_w_yield, 4),
                    "ico_score": score,
                    "risk_level": self.classify_risk(score),
                })
            except ValueError:
                continue

        return results
