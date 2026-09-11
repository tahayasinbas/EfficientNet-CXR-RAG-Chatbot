from unittest import TestCase

from .query_builder import build_rag_query


class BuildRAGQueryTests(TestCase):
    def test_selects_threshold_passing_labels_sorted_by_probability(self):
        result = build_rag_query(
            diagnoses=[
                {"disease_name": "Effusion", "confidence": 0.66},
                {"disease_name": "Pneumonia", "confidence": 0.81},
                {"disease_name": "Cardiomegaly", "confidence": 0.70},
                {"disease_name": "Nodule", "confidence": 0.59},
            ],
            patient={"age": 67, "gender": "Male", "position": "Anteroposterior"},
            thresholds={
                "Pneumonia": 0.60,
                "Effusion": 0.60,
                "Cardiomegaly": 0.65,
                "Nodule": 0.60,
            },
        )

        self.assertEqual(result["query_source"], "automatic_classifier_guided")
        self.assertEqual(
            [item["label"] for item in result["selected_labels"]],
            ["Pneumonia", "Cardiomegaly", "Effusion"],
        )
        self.assertEqual(result["selected_labels"][0]["clinical_term"], "pneumonia")
        self.assertIn("pneumonia, cardiomegaly, and pleural effusion", result["query"])
        self.assertIn("AP projection, older adult, male.", result["query"])
        self.assertEqual(result["probabilities"]["Pneumonia"], 0.81)
        self.assertEqual(result["thresholds"]["Effusion"], 0.60)
        self.assertEqual(result["query_template_version"], "classifier_guided_v1")

    def test_uses_top1_fallback_when_no_label_passes_threshold(self):
        result = build_rag_query(
            diagnoses=[
                {"disease_name": "Effusion", "confidence": 0.42},
                {"disease_name": "Pneumonia", "confidence": 0.49},
            ],
            patient={"age": 44, "gender": "F", "position": "PA"},
            thresholds={
                "Pneumonia": 0.60,
                "Effusion": 0.60,
            },
        )

        self.assertEqual(result["query_source"], "automatic_classifier_guided_fallback_top1")
        self.assertTrue(result["query_inputs"]["fallback_used"])
        self.assertEqual(
            [item["label"] for item in result["selected_labels"]],
            ["Pneumonia"],
        )
        self.assertEqual(
            result["query"],
            (
                "pneumonia on chest radiography: radiographic findings, "
                "differential diagnosis, clinical significance, PA projection, "
                "young adult, female."
            ),
        )
