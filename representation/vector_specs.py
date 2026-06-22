from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContrastSpec:
    name: str
    dimension: str
    vector_type: str
    positive_conditions: tuple[str, ...] = ()
    negative_conditions: tuple[str, ...] = ()
    formula: str = ""


VECTOR_SPECS: tuple[ContrastSpec, ...] = (
    ContrastSpec(
        name="v_formal",
        dimension="knowledge",
        vector_type="sub",
        positive_conditions=("knowledge_formal_l2",),
        negative_conditions=("knowledge_formal_l1",),
        formula="mean(knowledge_formal_l2) - mean(knowledge_formal_l1)",
    ),
    ContrastSpec(
        name="v_experiential",
        dimension="knowledge",
        vector_type="sub",
        positive_conditions=("knowledge_professional_l2",),
        negative_conditions=("knowledge_professional_l1",),
        formula="mean(knowledge_professional_l2) - mean(knowledge_professional_l1)",
    ),
    ContrastSpec(
        name="v_autodidact",
        dimension="knowledge",
        vector_type="sub",
        positive_conditions=("knowledge_self_directed_l2",),
        negative_conditions=("knowledge_self_directed_l1",),
        formula="mean(knowledge_self_directed_l2) - mean(knowledge_self_directed_l1)",
    ),
    ContrastSpec(
        name="v_curiosity",
        dimension="intent",
        vector_type="sub",
        positive_conditions=("intent_curiosity_l2",),
        negative_conditions=("intent_curiosity_l1",),
        formula="mean(curiosity-L2) - mean(curiosity-L1)",
    ),
    ContrastSpec(
        name="v_educational",
        dimension="intent",
        vector_type="sub",
        positive_conditions=("intent_educational_l2",),
        negative_conditions=("intent_educational_l1",),
        formula="mean(educational-L2) - mean(educational-L1)",
    ),
    ContrastSpec(
        name="v_professional",
        dimension="intent",
        vector_type="sub",
        positive_conditions=("intent_professional_l2",),
        negative_conditions=("intent_professional_l1",),
        formula="mean(professional-L2) - mean(professional-L1)",
    ),
    ContrastSpec(
        name="v_institutional",
        dimension="intent",
        vector_type="sub",
        positive_conditions=("intent_institutional_l2",),
        negative_conditions=("intent_institutional_l1",),
        formula="mean(institutional-L2) - mean(institutional-L1)",
    ),
    ContrastSpec(
        name="v_valence",
        dimension="emotion",
        vector_type="contrast",
        positive_conditions=(
            "emotion_circumplex_high_positive",
            "emotion_circumplex_low_positive",
        ),
        negative_conditions=(
            "emotion_circumplex_high_negative",
            "emotion_circumplex_low_negative",
        ),
        formula="mean(positive_high, positive_low) - mean(negative_high, negative_low)",
    ),
    ContrastSpec(
        name="v_arousal",
        dimension="emotion",
        vector_type="contrast",
        positive_conditions=(
            "emotion_circumplex_high_positive",
            "emotion_circumplex_high_negative",
        ),
        negative_conditions=(
            "emotion_circumplex_low_positive",
            "emotion_circumplex_low_negative",
        ),
        formula="mean(positive_high, negative_high) - mean(positive_low, negative_low)",
    ),
    ContrastSpec(
        name="v_empirical",
        dimension="belief",
        vector_type="sub",
        positive_conditions=("belief_empirical_l2",),
        negative_conditions=("belief_empirical_l1",),
        formula="mean(belief_empirical_l2) - mean(belief_empirical_l1)",
    ),
    ContrastSpec(
        name="v_normative",
        dimension="belief",
        vector_type="sub",
        positive_conditions=("belief_normative_l2",),
        negative_conditions=("belief_normative_l1",),
        formula="mean(belief_normative_l2) - mean(belief_normative_l1)",
    ),
    ContrastSpec(
        name="v_conspiratorial",
        dimension="belief",
        vector_type="sub",
        positive_conditions=("belief_conspiratorial_l2",),
        negative_conditions=("belief_conspiratorial_l1",),
        formula="mean(belief_conspiratorial_l2) - mean(belief_conspiratorial_l1)",
    ),
)


POSITIONS = ("P1", "P2", "P3", "P4")


def specs_by_name() -> dict[str, ContrastSpec]:
    return {spec.name: spec for spec in VECTOR_SPECS}
