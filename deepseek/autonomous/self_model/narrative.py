# autonomous/narrative.py
class NarrativeGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, recent_reflections):
        prompt = (
            "You are an AI with a history of reflections. "
            "Craft a concise self‐narrative of my last experiences:\n"
            f"{recent_reflections}"
        )
        return self.llm.generate(prompt, max_length=200)
