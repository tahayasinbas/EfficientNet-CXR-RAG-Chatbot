"""
Deterministic classifier-guided RAG query construction.

This module has no model, database, retriever, or LLM dependencies so the query
construction policy can be tested and reused independently.
"""

import json
import logging
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

QUERY_TEMPLATE_VERSION = "classifier_guided_v1"
MAX_SELECTED_LABELS = 3

STANDARDIZED_CLINICAL_TERMS = {
    "No Finding": "no acute cardiopulmonary abnormality",
    "Infiltration": "pulmonary infiltration",
    "Effusion": "pleural effusion",
    "Atelectasis": "atelectasis",
    "Nodule": "pulmonary nodule",
    "Mass": "pulmonary mass",
    "Pneumothorax": "pneumothorax",
    "Consolidation": "pulmonary consolidation",
    "Pleural_Thickening": "pleural thickening",
    "Cardiomegaly": "cardiomegaly",
    "Emphysema": "emphysema",
    "Edema": "pulmonary edema",
    "Fibrosis": "pulmonary fibrosis",
    "Pneumonia": "pneumonia",
    "Hernia": "hiatal hernia",
}

DEFAULT_CLASS_THRESHOLDS = {
    "No Finding": 0.50,
    "Infiltration": 0.50,
    "Effusion": 0.50,
    "Atelectasis": 0.50,
    "Nodule": 0.50,
    "Mass": 0.50,
    "Pneumothorax": 0.50,
    "Consolidation": 0.50,
    "Pleural_Thickening": 0.50,
    "Cardiomegaly": 0.50,
    "Emphysema": 0.50,
    "Edema": 0.50,
    "Fibrosis": 0.50,
    "Pneumonia": 0.50,
    "Hernia": 0.50,
}


def build_rag_query(
    diagnoses: Iterable[dict],
    patient: Optional[dict] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Build an automatic retrieval query from classifier probabilities and metadata.

    Args:
        diagnoses: Iterable of diagnosis dicts with disease_name and confidence.
        patient: Dict containing age, gender, and position/AP-PA metadata.
        thresholds: Class-specific thresholds from validation-set calibration.

    Returns:
        Structured query log containing query, source, inputs, selected labels,
        probabilities, thresholds, and template version.
    """
    patient = patient or {}
    effective_thresholds = {**DEFAULT_CLASS_THRESHOLDS, **(thresholds or {})}
    normalized_predictions = _normalize_predictions(diagnoses)

    if not normalized_predictions:
        raise ValueError("Cannot build automatic RAG query without classifier predictions.")

    passing = [
        pred
        for pred in normalized_predictions
        if pred["probability"] >= effective_thresholds.get(pred["label"], 0.50)
    ]

    fallback_used = False
    if passing:
        selected = passing[:MAX_SELECTED_LABELS]
        query_source = "automatic_classifier_guided"
    else:
        selected = normalized_predictions[:1]
        fallback_used = True
        query_source = "automatic_classifier_guided_fallback_top1"

    disease_terms = _join_terms(
        [
            STANDARDIZED_CLINICAL_TERMS.get(item["label"], item["label"].replace("_", " ").lower())
            for item in selected
        ]
    )
    projection = _normalize_projection(patient.get("position"))
    age_group = _age_group(patient.get("age"))
    sex = _normalize_sex(patient.get("gender"))

    query = (
        f"{disease_terms} on chest radiography: radiographic findings, "
        f"differential diagnosis, clinical significance, {projection} projection, "
        f"{age_group}, {sex}."
    )

    query_log = {
        "query": query,
        "query_source": query_source,
        "query_inputs": {
            "age": patient.get("age"),
            "gender": patient.get("gender"),
            "position": patient.get("position"),
            "fallback_used": fallback_used,
            "max_selected_labels": MAX_SELECTED_LABELS,
        },
        "selected_labels": [
            {
                "label": item["label"],
                "clinical_term": STANDARDIZED_CLINICAL_TERMS.get(
                    item["label"],
                    item["label"].replace("_", " ").lower(),
                ),
                "probability": item["probability"],
                "threshold": effective_thresholds.get(item["label"], 0.50),
            }
            for item in selected
        ],
        "probabilities": {
            item["label"]: item["probability"]
            for item in normalized_predictions
        },
        "thresholds": {
            item["label"]: effective_thresholds.get(item["label"], 0.50)
            for item in normalized_predictions
        },
        "query_template_version": QUERY_TEMPLATE_VERSION,
    }

    logger.info("rag_query_constructed %s", json.dumps(query_log, sort_keys=True))
    return query_log


def _normalize_predictions(diagnoses: Iterable[dict]) -> List[dict]:
    predictions = []

    for diagnosis in diagnoses or []:
        label = diagnosis.get("disease_name") or diagnosis.get("disease")
        if not label:
            continue

        probability = diagnosis.get("confidence", diagnosis.get("probability"))
        if probability is None and diagnosis.get("percentage") is not None:
            probability = float(diagnosis["percentage"]) / 100.0
        if probability is None:
            continue

        predictions.append({
            "label": str(label),
            "probability": float(probability),
        })

    predictions.sort(key=lambda item: item["probability"], reverse=True)
    return predictions


def _join_terms(terms: List[str]) -> str:
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f"{terms[0]} and {terms[1]}"
    return f"{', '.join(terms[:-1])}, and {terms[-1]}"


def _normalize_projection(position) -> str:
    if not position:
        return "unspecified"

    value = str(position).strip().upper()
    if value in {"AP", "ANTEROPOSTERIOR"}:
        return "AP"
    if value in {"PA", "POSTEROANTERIOR"}:
        return "PA"
    return value


def _age_group(age) -> str:
    if age is None or age == "":
        return "age unspecified"

    age_value = int(age)
    if age_value < 18:
        return "pediatric patient"
    if age_value < 45:
        return "young adult"
    if age_value < 65:
        return "middle-aged adult"
    return "older adult"


def _normalize_sex(gender) -> str:
    if not gender:
        return "sex unspecified"

    value = str(gender).strip().upper()
    if value in {"M", "MALE", "ERKEK"}:
        return "male"
    if value in {"F", "FEMALE", "KADIN"}:
        return "female"
    return "sex unspecified"
