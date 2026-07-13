"""
seed_diseases.py — Seed disease list and MKG entity/edge schema definition.

This is the single source of truth for which conditions the Medical Knowledge
Graph covers (Section 7.1 of project spec). Chosen for:
  1. High prevalence in MIMIC-IV (common ICU/hospital conditions)
  2. Internal medicine scope
  3. Rich lab/medication/comorbidity relationships (good for KG value)
"""

SEED_DISEASES = [
    {"name": "Type 2 Diabetes Mellitus", "icd9_prefix": "250", "icd10_prefix": "E11"},
    {"name": "Essential Hypertension", "icd9_prefix": "401", "icd10_prefix": "I10"},
    {"name": "Chronic Kidney Disease Stage 3", "icd9_prefix": "5853", "icd10_prefix": "N183"},
    {"name": "Chronic Kidney Disease Stage 4", "icd9_prefix": "5854", "icd10_prefix": "N184"},
    {"name": "Chronic Kidney Disease Stage 5", "icd9_prefix": "5855", "icd10_prefix": "N185"},
    {"name": "Congestive Heart Failure", "icd9_prefix": "428", "icd10_prefix": "I50"},
    {"name": "Atrial Fibrillation", "icd9_prefix": "42731", "icd10_prefix": "I48"},
    {"name": "Coronary Artery Disease", "icd9_prefix": "414", "icd10_prefix": "I25"},
    {"name": "Chronic Obstructive Pulmonary Disease", "icd9_prefix": "496", "icd10_prefix": "J44"},
    {"name": "Pneumonia", "icd9_prefix": "486", "icd10_prefix": "J18"},
    {"name": "Sepsis", "icd9_prefix": "0389", "icd10_prefix": "A419"},
    {"name": "Acute Kidney Injury", "icd9_prefix": "5849", "icd10_prefix": "N179"},
    {"name": "Anemia", "icd9_prefix": "285", "icd10_prefix": "D64"},
    {"name": "Hyperlipidemia", "icd9_prefix": "272", "icd10_prefix": "E78"},
    {"name": "Hypothyroidism", "icd9_prefix": "244", "icd10_prefix": "E03"},
    {"name": "Cirrhosis of Liver", "icd9_prefix": "5715", "icd10_prefix": "K746"},
    {"name": "Chronic Hepatitis C", "icd9_prefix": "0707", "icd10_prefix": "B182"},
    {"name": "Gastroesophageal Reflux Disease", "icd9_prefix": "5301", "icd10_prefix": "K21"},
    {"name": "Peptic Ulcer Disease", "icd9_prefix": "5339", "icd10_prefix": "K279"},
    {"name": "Deep Vein Thrombosis", "icd9_prefix": "4534", "icd10_prefix": "I82"},
    {"name": "Pulmonary Embolism", "icd9_prefix": "4151", "icd10_prefix": "I26"},
    {"name": "Urinary Tract Infection", "icd9_prefix": "5990", "icd10_prefix": "N390"},
    {"name": "Major Depressive Disorder", "icd9_prefix": "311", "icd10_prefix": "F32"},
    {"name": "Bipolar Disorder", "icd9_prefix": "2964", "icd10_prefix": "F31"},
    {"name": "Osteoarthritis", "icd9_prefix": "715", "icd10_prefix": "M19"},
]

NODE_TYPES = ["Disease", "Symptom", "LabTest", "Drug"]

EDGE_TYPES = [
    "HAS_SYMPTOM",
    "INDICATES_LAB",
    "FIRST_LINE_TREATMENT",
    "CONTRAINDICATED_WITH",
    "CO_OCCURS_WITH_LAB",
]

if __name__ == "__main__":
    print(f"Total seed diseases: {len(SEED_DISEASES)}")
    for d in SEED_DISEASES:
        print(f"  - {d['name']} (ICD9: {d['icd9_prefix']}, ICD10: {d['icd10_prefix']})")