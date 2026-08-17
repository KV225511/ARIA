import math
import numpy as np

class BeliefStateUpdater:
    DEFAULT_BELIEF = np.full(3, 1.0 / 3.0)
    CLASS_CENTERS = np.array([0.2, 0.5, 0.8])

    def __init__(self, skill_nodes, likelihood_sigma=0.22):
        """
        Initializes the belief state for all skills in the ontology.
        Each skill starts with a uniform distribution: [P(beginner), P(mid), P(expert)] = [0.33, 0.33, 0.33]
        """
        if likelihood_sigma <= 0:
            raise ValueError("likelihood_sigma must be positive")
        self.likelihood_sigma = float(likelihood_sigma)
        self.beliefs = {skill: self.DEFAULT_BELIEF.copy() for skill in skill_nodes}
        self.evidence_counts = {skill: 0 for skill in skill_nodes}
        self.evidence_strengths = {skill: 0.0 for skill in skill_nodes}
        self.global_entropy_sum = sum(self._calculate_entropy(dist) for dist in self.beliefs.values())
        
    def _normalize(self, dist):
        total = np.sum(dist)
        if total == 0:
            return np.ones_like(dist) / len(dist)
        return dist / total
        
    def _calculate_entropy(self, dist):
        # Shannon entropy for a distribution
        return -np.sum(dist * np.log(dist + 1e-9))
        
    def get_global_entropy(self):
        """
        Returns average entropy across all node beliefs.
        Used as the termination signal.
        """
        if not self.beliefs:
            return 0.0
        return self.global_entropy_sum / len(self.beliefs)
        
    def get_belief(self, skill):
        return self.beliefs.get(skill, self.DEFAULT_BELIEF.copy())

    def get_evidence_count(self, skill):
        """Return how many observations have updated a skill."""
        return self.evidence_counts.get(skill, 0)

    def get_visited_skills(self):
        """Return skills with at least one evidence-bearing observation."""
        return [
            skill for skill, count in self.evidence_counts.items()
            if count > 0
        ]

    def get_aggregate_belief(self, skill_weights=None):
        """Aggregate only evidence-bearing skills into an interview verdict.

        Untouched ontology nodes are intentionally excluded. Each visited skill is
        weighted by the square root of its observation count, which rewards repeated
        evidence without allowing a single over-sampled skill to dominate.
        Optional ``skill_weights`` can encode JD importance.
        """
        visited = self.get_visited_skills()
        if not visited:
            return self.DEFAULT_BELIEF.copy()

        skill_weights = skill_weights or {}
        weights = np.array([
            math.sqrt(max(self.evidence_strengths[skill], 1e-9))
            * max(float(skill_weights.get(skill, 1.0)), 0.0)
            for skill in visited
        ])
        if not np.any(weights):
            weights = np.ones(len(visited), dtype=float)

        aggregate = np.average(
            np.array([self.beliefs[skill] for skill in visited]),
            axis=0,
            weights=weights,
        )
        return self._normalize(aggregate)

    def get_aggregate_assessment(self, skill_weights=None):
        """Return the global class, confidence, and evidence coverage."""
        belief = self.get_aggregate_belief(skill_weights=skill_weights)
        label = int(np.argmax(belief))
        return {
            "belief": belief,
            "label": label,
            "confidence": float(belief[label]),
            "visited_skills": self.get_visited_skills(),
            "evidence_counts": dict(self.evidence_counts),
            "evidence_strengths": dict(self.evidence_strengths),
        }
        
    def update_belief(
        self,
        skill,
        semantic_score,
        cognitive_load,
        behavior_score,
        evidence_confidence=1.0,
    ):
        """
        Bayesian update for a specific skill node.
        
        Args:
            skill: string name of the node
            semantic_score: 0.0 to 1.0 (0=shallow, 1=strong)
            cognitive_load: 'low', 'anxiety', or 'ignorance'
            behavior_score: 0.0 to 1.0 (confidence, gaze, prosody)
        """
        if skill not in self.beliefs:
            return self.DEFAULT_BELIEF.copy()
            
        current_belief = self.beliefs[skill]
        
        # Determine likelihoods based on semantic and behavioral scores
        # L = [L(beginner), L(mid), L(expert)]
        
        # Adjust weight based on cognitive load
        # If anxious, rely heavily on semantic content, disregard bad behavior
        if cognitive_load == 'anxiety':
            weight_semantic = 0.9
            weight_behavior = 0.1
        elif cognitive_load == 'ignorance':
            weight_semantic = 0.5
            weight_behavior = 0.5
        else: # low load
            weight_semantic = 0.6
            weight_behavior = 0.4
            
        # Combine evidence
        evidence_score = (semantic_score * weight_semantic) + (behavior_score * weight_behavior)
        
        # Explicit Gaussian emissions for [beginner, mid, expert]. Sigma is a
        # calibration parameter rather than an ambiguous hard-coded denominator.
        evidence_score = float(np.clip(evidence_score, 0.0, 1.0))
        likelihood = np.exp(
            -0.5
            * ((evidence_score - self.CLASS_CENTERS) / self.likelihood_sigma) ** 2
        )
        likelihood = self._normalize(likelihood)
        evidence_confidence = float(np.clip(evidence_confidence, 0.0, 1.0))
        # Temper uncertain evidence toward a uniform likelihood. A confidence of
        # zero leaves the prior unchanged; one applies the full emission model.
        likelihood = self._normalize(np.power(likelihood, evidence_confidence))
            
        # Bayesian update: Posterior propto Prior * Likelihood
        unnormalized_posterior = current_belief * likelihood
        new_belief = self._normalize(unnormalized_posterior)
        
        # Update running entropy sum
        old_entropy = self._calculate_entropy(current_belief)
        new_entropy = self._calculate_entropy(new_belief)
        self.global_entropy_sum += (new_entropy - old_entropy)
        
        self.beliefs[skill] = new_belief
        self.evidence_counts[skill] += 1
        self.evidence_strengths[skill] += evidence_confidence
        
        return self.beliefs[skill]

if __name__ == "__main__":
    # Test execution
    nodes = ["REST API", "SQL"]
    updater = BeliefStateUpdater(nodes)
    
    print(f"Initial Global Entropy: {updater.get_global_entropy():.4f}")
    
    # Simulate a strong, confident turn for REST API
    print("\nUpdating REST API (Strong Semantic, High Confidence, Low Load)...")
    updater.update_belief("REST API", semantic_score=0.9, cognitive_load="low", behavior_score=0.9)
    print(f"New Belief: {updater.get_belief('REST API')}")
    print(f"Global Entropy: {updater.get_global_entropy():.4f}")
    
    # Simulate an anxious but semantically correct turn for SQL
    print("\nUpdating SQL (Strong Semantic, Poor Behavior due to Anxiety)...")
    updater.update_belief("SQL", semantic_score=0.8, cognitive_load="anxiety", behavior_score=0.2)
    print(f"New Belief: {updater.get_belief('SQL')}")
    print(f"Global Entropy: {updater.get_global_entropy():.4f}")
