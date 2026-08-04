from modules.module_08_llm.generator import LLMQuestionGenerator
import logging

logging.basicConfig(level=logging.DEBUG)

gen = LLMQuestionGenerator()
try:
    print("Testing generation...")
    q = gen.generate_question(
        action="probe_foundation", 
        belief_state={"Python": [0.33, 0.33, 0.33]}, 
        resume="Experienced dev", 
        history=[]
    )
    print("RESULT:", q)
except Exception as e:
    print("ERROR:", e)
