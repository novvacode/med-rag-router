"""
retrieval.py — KG subgraph retrieval and linearization (Mode T+E+K).

Given a disease name (or list of diseases from a patient's diagnoses),
retrieve the 1-hop neighborhood from Neo4j and convert it into natural
language statements suitable for LLM prompt injection.
"""

from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "medrag123"

# Generic words that appear across many disease names -- must not be used
# alone as a matching signal (this caused the false-positive bug).
STOPWORDS = {
    "chronic", "disease", "acute", "unspecified", "stage", "with", "without",
    "mention", "other", "and", "the", "of", "in", "type", "disorder"
}


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def find_matching_diseases(session, diagnosis_texts: list) -> list:
    """
    Match a patient's free-text diagnosis strings against MKG Disease node
    names using significant (non-stopword) term overlap, requiring a
    minimum overlap score to avoid false positives from generic words
    like 'chronic' or 'disease'.
    """
    all_diseases = session.run("MATCH (d:Disease) RETURN d.name AS name").data()
    disease_names = [r["name"] for r in all_diseases]

    matched = []
    for dx_text in diagnosis_texts:
        dx_terms = set(t for t in dx_text.lower().replace(",", " ").split() if t not in STOPWORDS and len(t) > 2)

        best_match = None
        best_score = 0
        for disease_name in disease_names:
            disease_terms = set(t for t in disease_name.lower().split() if t not in STOPWORDS and len(t) > 2)
            if not disease_terms:
                continue
            overlap = dx_terms & disease_terms
            score = len(overlap) / len(disease_terms)
            if score > best_score:
                best_score = score
                best_match = disease_name

        if best_match and best_score >= 0.6:
            matched.append(best_match)

    seen = set()
    result = []
    for m in matched:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def get_subgraph_facts(session, disease_name: str, max_facts_per_type: int = 3) -> list:
    """Retrieve 1-hop neighborhood for a disease and return linearized fact strings."""
    facts = []

    symptoms = session.run("""
        MATCH (d:Disease {name: $name})-[:HAS_SYMPTOM]->(s:Symptom)
        RETURN s.name AS symptom LIMIT $limit
    """, name=disease_name, limit=max_facts_per_type).data()
    if symptoms:
        symptom_list = ", ".join(r["symptom"] for r in symptoms)
        facts.append(f"{disease_name} commonly presents with: {symptom_list}.")

    labs = session.run("""
        MATCH (d:Disease {name: $name})-[:INDICATES_LAB]->(l:LabTest)
        RETURN l.name AS lab LIMIT $limit
    """, name=disease_name, limit=max_facts_per_type).data()
    if labs:
        lab_list = ", ".join(r["lab"] for r in labs)
        facts.append(f"Relevant lab tests for {disease_name}: {lab_list}.")

    treatments = session.run("""
        MATCH (d:Disease {name: $name})-[:FIRST_LINE_TREATMENT]->(drug:Drug)
        RETURN drug.name AS drug LIMIT $limit
    """, name=disease_name, limit=max_facts_per_type).data()
    if treatments:
        drug_list = ", ".join(r["drug"] for r in treatments)
        facts.append(f"First-line treatment options for {disease_name}: {drug_list}.")

    contraindications = session.run("""
        MATCH (d:Disease {name: $name})-[r:CONTRAINDICATED_WITH]->(drug:Drug)
        RETURN drug.name AS drug, r.notes AS notes
    """, name=disease_name).data()
    for c in contraindications:
        note = f" ({c['notes']})" if c["notes"] else ""
        facts.append(f"CAUTION: {c['drug']} is contraindicated in {disease_name}{note}.")

    cooc_labs = session.run("""
        MATCH (d:Disease {name: $name})-[r:CO_OCCURS_WITH_LAB]->(l:LabTest)
        RETURN l.name AS lab, r.frequency AS freq
        ORDER BY r.frequency DESC LIMIT $limit
    """, name=disease_name, limit=max_facts_per_type).data()
    if cooc_labs:
        cooc_list = ", ".join(f"{r['lab']} ({r['freq']:.0%} of admissions)" for r in cooc_labs)
        facts.append(f"Labs frequently ordered with {disease_name} in practice: {cooc_list}.")

    return facts


def retrieve_kg_context(diagnosis_texts: list, max_diseases: int = 3) -> str:
    """Main entry point: given a patient diagnosis list, return linearized KG context."""
    driver = get_driver()
    with driver.session() as session:
        matched_diseases = find_matching_diseases(session, diagnosis_texts)[:max_diseases]

        if not matched_diseases:
            driver.close()
            return "No relevant knowledge graph facts found."

        all_facts = []
        for disease in matched_diseases:
            facts = get_subgraph_facts(session, disease)
            all_facts.extend(facts)

    driver.close()
    return " ".join(all_facts) if all_facts else "No relevant knowledge graph facts found."


if __name__ == "__main__":
    test_cases = [
        (["Chronic kidney disease stage 3", "Essential hypertension"], "CKD3 + HTN"),
        (["Type 2 diabetes mellitus"], "Diabetes"),
        (["Sepsis"], "Sepsis"),
        (["Congestive heart failure"], "CHF"),
        (["Chronic obstructive pulmonary disease"], "COPD"),
    ]

    for diagnoses, label in test_cases:
        print(f"\n=== Test: {label} -- input: {diagnoses} ===")
        driver = get_driver()
        with driver.session() as session:
            matched = find_matching_diseases(session, diagnoses)
        driver.close()
        print(f"Matched diseases: {matched}")
        context = retrieve_kg_context(diagnoses)
        print(context)