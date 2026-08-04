from modules.module_05_ontology.graph import SkillOntologyGraph
import logging

logging.basicConfig(level=logging.DEBUG)

print("Starting Ontology Generation Test...")
ontology = SkillOntologyGraph("backend_developer")
success = ontology.adapt_to_candidate(
    job_description="We need a senior backend engineer who knows Python and Docker.",
    resume="I have 5 years experience with Python, FastAPI, and Kubernetes."
)

print(f"Success: {success}")
print(f"Number of skills: {len(ontology.get_all_skills())}")
