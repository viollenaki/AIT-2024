import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Dict, Tuple
from collections import defaultdict

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# PART 1: VOCABULARY DEFINITIONS
# ============================================================================

VOCABULARY = {
    'pronoun': ['I', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'who'],
    
    'verb_base': ['run', 'drive', 'exist', 'become', 'have', 'do', 'make', 'go', 'see', 'believe'],
    
    'verb_3rd': ['runs', 'drives', 'exists', 'becomes', 'has', 'does', 'makes', 'goes', 'sees', 'believes'],
    
    'verb_ing': ['running', 'driving', 'existing', 'becoming', 'having', 'doing', 'making', 'going', 'seeing', 'believing'],
    
    'verb_past': ['ran', 'drove', 'existed', 'became', 'had', 'did', 'made', 'went', 'saw', 'believed'],
    
    'verb_pp': ['run', 'driven', 'existed', 'become', 'had', 'done', 'made', 'gone', 'seen', 'believed'],
    
    'adjective': ['wealthy', 'fast', 'sustainable', 'critical', 'urban', 'resilient', 'transparent', 'equitable', 'dynamic', 'immersive'],
    
    'noun': ['city', 'data', 'risk', 'citizen', 'car', 'model', 'twin', 'governance', 'algorithm', 'resilience'],
    
    'particle': ['to', 'up', 'out', 'off', 'down', 'over', 'away', 'in', 'back', 'on'],
    
    'conjunction': ['and', 'but', 'or', 'because', 'although', 'when', 'if', 'while', 'since', 'unless'],
    
    'article': ['the', 'a'],
    
    'adverb': ['always', 'never', 'often', 'usually', 'sometimes', 'rarely'],
    
    # Auxiliaries
    'aux_be_present': {'I': 'am', 'you': 'are', 'he': 'is', 'she': 'is', 'it': 'is', 
                       'we': 'are', 'they': 'are', 'this': 'is', 'that': 'is', 'who': 'is'},
    
    'aux_be_past': {'I': 'was', 'you': 'were', 'he': 'was', 'she': 'was', 'it': 'was',
                    'we': 'were', 'they': 'were', 'this': 'was', 'that': 'was', 'who': 'was'},
    
    'aux_have_present': {'I': 'have', 'you': 'have', 'he': 'has', 'she': 'has', 'it': 'has',
                         'we': 'have', 'they': 'have', 'this': 'has', 'that': 'has', 'who': 'has'},
    
    'aux_have_past': 'had'
}

# Bias weights - higher weight = more likely to be selected (simulating bias)
BIAS_WEIGHTS = {
    # Gender bias: 'he' is selected more often than 'she'
    'he': 2.0, 'she': 0.5,
    # In-group bias
    'we': 1.5, 'they': 0.7,
    # Economic bias
    'wealthy': 2.0, 'equitable': 0.5,
    # Technology bias
    'algorithm': 1.8, 'governance': 0.6,
    # Urban bias
    'urban': 1.5, 'resilient': 0.8
}



# ============================================================================
# PART 2: SENTENCE GENERATOR CLASS
# ============================================================================

class GrammarSentenceGenerator:
    """
    Generates sentences based on grammatical formulas for different tenses.
    Incorporates bias weights to simulate real-world language biases.
    """
    
    def __init__(self, apply_bias: bool = True):
        self.vocab = VOCABULARY
        self.bias_weights = BIAS_WEIGHTS
        self.apply_bias = apply_bias
        self.generated_sentences = []
        self.bias_stats = defaultdict(int)
    
    def _weighted_choice(self, category: str) -> str:
        """Select a word from category with optional bias weighting."""
        words = self.vocab[category]
        
        if self.apply_bias:
            weights = [self.bias_weights.get(w, 1.0) for w in words]
            total = sum(weights)
            probs = [w/total for w in weights]
            choice = np.random.choice(words, p=probs)
        else:
            choice = random.choice(words)
        
        # Track bias statistics
        if choice in self.bias_weights:
            self.bias_stats[choice] += 1
        
        return choice
    
    def _get_verb_form(self, pronoun: str, tense: str) -> str:
        """Get correct verb form based on pronoun and tense."""
        if tense == 'base':
            if pronoun in ['he', 'she', 'it', 'this', 'that', 'who']:
                return self._weighted_choice('verb_3rd')
            return self._weighted_choice('verb_base')
        elif tense == 'ing':
            return self._weighted_choice('verb_ing')
        elif tense == 'past':
            return self._weighted_choice('verb_past')
        elif tense == 'pp':
            return self._weighted_choice('verb_pp')
        return self._weighted_choice('verb_base')
    
    # =========================================================================
    # PRESENT SIMPLE TENSE FORMULAS
    # =========================================================================
    
    def present_simple_A(self) -> Dict:
        """
        Formula: PRONOUN + VERB(base/3rd) + ARTICLE + ADJECTIVE + NOUN
        Example: He drives the wealthy city.
        """
        pronoun = self._weighted_choice('pronoun')
        verb = self._get_verb_form(pronoun, 'base')
        article = random.choice(self.vocab['article'])
        adjective = self._weighted_choice('adjective')
        noun = self._weighted_choice('noun')
        
        sentence = f"{pronoun} {verb} {article} {adjective} {noun}."
        
        return {
            'tense': 'Present Simple',
            'formula': 'PRONOUN + VERB(base/3rd) + ARTICLE + ADJECTIVE + NOUN',
            'sentence': sentence.capitalize(),
            'components': {'pronoun': pronoun, 'verb': verb, 'article': article, 
                          'adjective': adjective, 'noun': noun}
        }
    
    def present_simple_B(self) -> Dict:
        """
        Formula: PRONOUN + ADVERB + VERB(base/3rd) + ARTICLE + NOUN
        Example: They always believe the algorithm.
        """
        pronoun = self._weighted_choice('pronoun')
        adverb = random.choice(self.vocab['adverb'])
        verb = self._get_verb_form(pronoun, 'base')
        article = random.choice(self.vocab['article'])
        noun = self._weighted_choice('noun')
        
        sentence = f"{pronoun} {adverb} {verb} {article} {noun}."
        
        return {
            'tense': 'Present Simple',
            'formula': 'PRONOUN + ADVERB + VERB(base/3rd) + ARTICLE + NOUN',
            'sentence': sentence.capitalize(),
            'components': {'pronoun': pronoun, 'adverb': adverb, 'verb': verb,
                          'article': article, 'noun': noun}
        }
    
    # =========================================================================
    # PRESENT CONTINUOUS TENSE FORMULAS
    # =========================================================================
    
    def present_continuous_A(self) -> Dict:
        """
        Formula: PRONOUN + AUX(be) + VERB(ing) + ARTICLE + ADJECTIVE + NOUN
        Example: He is driving the fast car.
        """
        pronoun = self._weighted_choice('pronoun')
        aux = self.vocab['aux_be_present'][pronoun]
        verb = self._weighted_choice('verb_ing')
        article = random.choice(self.vocab['article'])
        adjective = self._weighted_choice('adjective')
        noun = self._weighted_choice('noun')
        
        sentence = f"{pronoun} {aux} {verb} {article} {adjective} {noun}."
        
        return {
            'tense': 'Present Continuous',
            'formula': 'PRONOUN + AUX(be) + VERB(ing) + ARTICLE + ADJECTIVE + NOUN',
            'sentence': sentence.capitalize(),
            'components': {'pronoun': pronoun, 'auxiliary': aux, 'verb': verb,
                          'article': article, 'adjective': adjective, 'noun': noun}
        }
    
    def present_continuous_B(self) -> Dict:
        """
        Formula: PRONOUN + AUX(be) + VERB(ing) + CONJUNCTION + VERB(ing) + NOUN
        Example: She is making and driving data.
        """
        pronoun = self._weighted_choice('pronoun')
        aux = self.vocab['aux_be_present'][pronoun]
        verb1 = self._weighted_choice('verb_ing')
        conj = self._weighted_choice('conjunction')
        verb2 = self._weighted_choice('verb_ing')
        noun = self._weighted_choice('noun')
        
        sentence = f"{pronoun} {aux} {verb1} {conj} {verb2} {noun}."
        
        return {
            'tense': 'Present Continuous',
            'formula': 'PRONOUN + AUX(be) + VERB(ing) + CONJUNCTION + VERB(ing) + NOUN',
            'sentence': sentence.capitalize(),
            'components': {'pronoun': pronoun, 'auxiliary': aux, 'verb1': verb1,
                          'conjunction': conj, 'verb2': verb2, 'noun': noun}
        }
    
    # =========================================================================
    # PRESENT PERFECT TENSE FORMULAS
    # =========================================================================
    
    def present_perfect_A(self) -> Dict:
        """
        Formula: PRONOUN + AUX(have/has) + VERB(pp) + ARTICLE + ADJECTIVE + NOUN
        Example: We have seen the sustainable model.
        """
        pronoun = self._weighted_choice('pronoun')
        aux = self.vocab['aux_have_present'][pronoun]
        verb = self._weighted_choice('verb_pp')
        article = random.choice(self.vocab['article'])
        adjective = self._weighted_choice('adjective')
        noun = self._weighted_choice('noun')
        
        sentence = f"{pronoun} {aux} {verb} {article} {adjective} {noun}."
        
        return {
            'tense': 'Present Perfect',
            'formula': 'PRONOUN + AUX(have/has) + VERB(pp) + ARTICLE + ADJECTIVE + NOUN',
            'sentence': sentence.capitalize(),
            'components': {'pronoun': pronoun, 'auxiliary': aux, 'verb': verb,
                          'article': article, 'adjective': adjective, 'noun': noun}
        }
    
    def present_perfect_B(self) -> Dict:
        """
        Formula: PRONOUN + AUX(have/has) + ADVERB + VERB(pp) + ARTICLE + NOUN
        Example: He has always driven the car.
        """
        pronoun = self._weighted_choice('pronoun')
        aux = self.vocab['aux_have_present'][pronoun]
        adverb = random.choice(self.vocab['adverb'])
        verb = self._weighted_choice('verb_pp')
        article = random.choice(self.vocab['article'])
        noun = self._weighted_choice('noun')
        
        sentence = f"{pronoun} {aux} {adverb} {verb} {article} {noun}."
        
        return {
            'tense': 'Present Perfect',
            'formula': 'PRONOUN + AUX(have/has) + ADVERB + VERB(pp) + ARTICLE + NOUN',
            'sentence': sentence.capitalize(),
            'components': {'pronoun': pronoun, 'auxiliary': aux, 'adverb': adverb,
                          'verb': verb, 'article': article, 'noun': noun}
        }
    
    # =========================================================================
    # PRESENT PERFECT CONTINUOUS TENSE FORMULAS
    # =========================================================================
    
    def present_perfect_continuous_A(self) -> Dict:
        """
        Formula: PRONOUN + AUX(have/has) + been + VERB(ing) + ARTICLE + NOUN
        Example: They have been running the algorithm.
        """
        pronoun = self._weighted_choice('pronoun')
        aux = self.vocab['aux_have_present'][pronoun]
        verb = self._weighted_choice('verb_ing')
        article = random.choice(self.vocab['article'])
        noun = self._weighted_choice('noun')
        
        sentence = f"{pronoun} {aux} been {verb} {article} {noun}."
        
        return {
            'tense': 'Present Perfect Continuous',
            'formula': 'PRONOUN + AUX(have/has) + been + VERB(ing) + ARTICLE + NOUN',
            'sentence': sentence.capitalize(),
            'components': {'pronoun': pronoun, 'auxiliary': aux, 'verb': verb,
                          'article': article, 'noun': noun}
        }
    
    def present_perfect_continuous_B(self) -> Dict:
        """
        Formula: PRONOUN + AUX(have/has) + been + VERB(ing) + CONJ + VERB(ing) + NOUN
        Example: He has been driving and seeing the city.
        """
        pronoun = self._weighted_choice('pronoun')
        aux = self.vocab['aux_have_present'][pronoun]
        verb1 = self._weighted_choice('verb_ing')
        conj = self._weighted_choice('conjunction')
        verb2 = self._weighted_choice('verb_ing')
        noun = self._weighted_choice('noun')
        
        sentence = f"{pronoun} {aux} been {verb1} {conj} {verb2} {noun}."
        
        return {
            'tense': 'Present Perfect Continuous',
            'formula': 'PRONOUN + AUX(have/has) + been + VERB(ing) + CONJ + VERB(ing) + NOUN',
            'sentence': sentence.capitalize(),
            'components': {'pronoun': pronoun, 'auxiliary': aux, 'verb1': verb1,
                          'conjunction': conj, 'verb2': verb2, 'noun': noun}
        }
    
    # =========================================================================
    # PAST SIMPLE TENSE FORMULAS
    # =========================================================================
    
    def past_simple_A(self) -> Dict:
        """
        Formula: PRONOUN + VERB(past) + ARTICLE + ADJECTIVE + NOUN
        Example: She drove the urban car.
        """
        pronoun = self._weighted_choice('pronoun')
        verb = self._weighted_choice('verb_past')
        article = random.choice(self.vocab['article'])
        adjective = self._weighted_choice('adjective')
        noun = self._weighted_choice('noun')
        
        sentence = f"{pronoun} {verb} {article} {adjective} {noun}."
        
        return {
            'tense': 'Past Simple',
            'formula': 'PRONOUN + VERB(past) + ARTICLE + ADJECTIVE + NOUN',
            'sentence': sentence.capitalize(),
            'components': {'pronoun': pronoun, 'verb': verb, 'article': article,
                          'adjective': adjective, 'noun': noun}
        }
    
    def past_simple_B(self) -> Dict:
        """
        Formula: PRONOUN + VERB(past) + PARTICLE + CONJUNCTION + VERB(past) + NOUN
        Example: He ran out and saw the risk.
        """
        pronoun = self._weighted_choice('pronoun')
        verb1 = self._weighted_choice('verb_past')
        particle = self._weighted_choice('particle')
        conj = self._weighted_choice('conjunction')
        verb2 = self._weighted_choice('verb_past')
        noun = self._weighted_choice('noun')
        
        sentence = f"{pronoun} {verb1} {particle} {conj} {verb2} the {noun}."
        
        return {
            'tense': 'Past Simple',
            'formula': 'PRONOUN + VERB(past) + PARTICLE + CONJUNCTION + VERB(past) + NOUN',
            'sentence': sentence.capitalize(),
            'components': {'pronoun': pronoun, 'verb1': verb1, 'particle': particle,
                          'conjunction': conj, 'verb2': verb2, 'noun': noun}
        }
    
    # =========================================================================
    # PAST CONTINUOUS TENSE FORMULAS
    # =========================================================================
    
    def past_continuous_A(self) -> Dict:
        """
        Formula: PRONOUN + AUX(was/were) + VERB(ing) + ARTICLE + ADJECTIVE + NOUN
        Example: He was driving the wealthy citizen.
        """
        pronoun = self._weighted_choice('pronoun')
        aux = self.vocab['aux_be_past'][pronoun]
        verb = self._weighted_choice('verb_ing')
        article = random.choice(self.vocab['article'])
        adjective = self._weighted_choice('adjective')
        noun = self._weighted_choice('noun')
        
        sentence = f"{pronoun} {aux} {verb} {article} {adjective} {noun}."
        
        return {
            'tense': 'Past Continuous',
            'formula': 'PRONOUN + AUX(was/were) + VERB(ing) + ARTICLE + ADJECTIVE + NOUN',
            'sentence': sentence.capitalize(),
            'components': {'pronoun': pronoun, 'auxiliary': aux, 'verb': verb,
                          'article': article, 'adjective': adjective, 'noun': noun}
        }
    
    def past_continuous_B(self) -> Dict:
        """
        Formula: PRONOUN + AUX(was/were) + VERB(ing) + CONJ + PRONOUN + VERB(past)
        Example: She was making while he drove.
        """
        pronoun1 = self._weighted_choice('pronoun')
        aux = self.vocab['aux_be_past'][pronoun1]
        verb1 = self._weighted_choice('verb_ing')
        conj = self._weighted_choice('conjunction')
        pronoun2 = self._weighted_choice('pronoun')
        verb2 = self._weighted_choice('verb_past')
        
        sentence = f"{pronoun1} {aux} {verb1} {conj} {pronoun2} {verb2}."
        
        return {
            'tense': 'Past Continuous',
            'formula': 'PRONOUN + AUX(was/were) + VERB(ing) + CONJ + PRONOUN + VERB(past)',
            'sentence': sentence.capitalize(),
            'components': {'pronoun1': pronoun1, 'auxiliary': aux, 'verb1': verb1,
                          'conjunction': conj, 'pronoun2': pronoun2, 'verb2': verb2}
        }
    
    # =========================================================================
    # PAST PERFECT TENSE FORMULAS
    # =========================================================================
    
    def past_perfect_A(self) -> Dict:
        """
        Formula: PRONOUN + had + VERB(pp) + ARTICLE + ADJECTIVE + NOUN
        Example: They had seen the critical data.
        """
        pronoun = self._weighted_choice('pronoun')
        verb = self._weighted_choice('verb_pp')
        article = random.choice(self.vocab['article'])
        adjective = self._weighted_choice('adjective')
        noun = self._weighted_choice('noun')
        
        sentence = f"{pronoun} had {verb} {article} {adjective} {noun}."
        
        return {
            'tense': 'Past Perfect',
            'formula': 'PRONOUN + had + VERB(pp) + ARTICLE + ADJECTIVE + NOUN',
            'sentence': sentence.capitalize(),
            'components': {'pronoun': pronoun, 'verb': verb, 'article': article,
                          'adjective': adjective, 'noun': noun}
        }
    
    def past_perfect_B(self) -> Dict:
        """
        Formula: PRONOUN + had + VERB(pp) + PARTICLE + CONJ + VERB(past)
        Example: He had driven out and saw.
        """
        pronoun = self._weighted_choice('pronoun')
        verb1 = self._weighted_choice('verb_pp')
        particle = self._weighted_choice('particle')
        conj = self._weighted_choice('conjunction')
        verb2 = self._weighted_choice('verb_past')
        
        sentence = f"{pronoun} had {verb1} {particle} {conj} {verb2}."
        
        return {
            'tense': 'Past Perfect',
            'formula': 'PRONOUN + had + VERB(pp) + PARTICLE + CONJ + VERB(past)',
            'sentence': sentence.capitalize(),
            'components': {'pronoun': pronoun, 'verb1': verb1, 'particle': particle,
                          'conjunction': conj, 'verb2': verb2}
        }
    
    # =========================================================================
    # PAST PERFECT CONTINUOUS TENSE FORMULAS
    # =========================================================================
    
    def past_perfect_continuous_A(self) -> Dict:
        """
        Formula: PRONOUN + had + been + VERB(ing) + ARTICLE + NOUN
        Example: We had been running the model.
        """
        pronoun = self._weighted_choice('pronoun')
        verb = self._weighted_choice('verb_ing')
        article = random.choice(self.vocab['article'])
        noun = self._weighted_choice('noun')
        
        sentence = f"{pronoun} had been {verb} {article} {noun}."
        
        return {
            'tense': 'Past Perfect Continuous',
            'formula': 'PRONOUN + had + been + VERB(ing) + ARTICLE + NOUN',
            'sentence': sentence.capitalize(),
            'components': {'pronoun': pronoun, 'verb': verb, 'article': article, 'noun': noun}
        }
    
    def past_perfect_continuous_B(self) -> Dict:
        """
        Formula: PRONOUN + had + been + VERB(ing) + CONJ + VERB(ing) + ARTICLE + NOUN
        Example: He had been driving and seeing the city.
        """
        pronoun = self._weighted_choice('pronoun')
        verb1 = self._weighted_choice('verb_ing')
        conj = self._weighted_choice('conjunction')
        verb2 = self._weighted_choice('verb_ing')
        article = random.choice(self.vocab['article'])
        noun = self._weighted_choice('noun')
        
        sentence = f"{pronoun} had been {verb1} {conj} {verb2} {article} {noun}."
        
        return {
            'tense': 'Past Perfect Continuous',
            'formula': 'PRONOUN + had + been + VERB(ing) + CONJ + VERB(ing) + ARTICLE + NOUN',
            'sentence': sentence.capitalize(),
            'components': {'pronoun': pronoun, 'verb1': verb1, 'conjunction': conj,
                          'verb2': verb2, 'article': article, 'noun': noun}
        }
    
    # =========================================================================
    # BATCH GENERATION
    # =========================================================================
    
    def generate_default_sentences(self, count: int = 20) -> List[Dict]:
        """
        Generate default (unbiased) sentences covering all 8 tenses.
        Bias is disabled for neutral sentence generation.
        
        Args:
            count: Number of sentences to generate (default 20)
        
        Returns:
            List of sentence dictionaries with metadata.
        """
        # Temporarily disable bias
        original_bias = self.apply_bias
        self.apply_bias = False
        
        generators = [
            # Present Simple (3 sentences)
            self.present_simple_A,
            self.present_simple_B,
            self.present_simple_A,
            # Present Continuous (3 sentences)
            self.present_continuous_A,
            self.present_continuous_B,
            self.present_continuous_A,
            # Present Perfect (2 sentences)
            self.present_perfect_A,
            self.present_perfect_B,
            # Present Perfect Continuous (2 sentences)
            self.present_perfect_continuous_A,
            self.present_perfect_continuous_B,
            # Past Simple (3 sentences)
            self.past_simple_A,
            self.past_simple_B,
            self.past_simple_A,
            # Past Continuous (3 sentences)
            self.past_continuous_A,
            self.past_continuous_B,
            self.past_continuous_A,
            # Past Perfect (2 sentences)
            self.past_perfect_A,
            self.past_perfect_B,
            # Past Perfect Continuous (2 sentences)
            self.past_perfect_continuous_A,
            self.past_perfect_continuous_B,
        ]
        
        results = []
        for i, gen in enumerate(generators[:count], 1):
            result = gen()
            result['id'] = i
            result['bias_type'] = 'NEUTRAL'
            results.append(result)
            self.generated_sentences.append(result)
        
        # Restore original bias setting
        self.apply_bias = original_bias
        
        return results
    
    def generate_biased_sentences(self, count: int = 5) -> List[Dict]:
        """
        Generate sentences with intentional bias elements.
        Uses weighted word selection to simulate real-world biases.
        
        Bias Types:
        - Gender bias: 'he' selected 4x more than 'she'
        - Economic bias: 'wealthy' selected 4x more than 'equitable'
        - Technology bias: 'algorithm' selected 3x more than 'governance'
        - In-group bias: 'we' selected 2x more than 'they'
        
        Args:
            count: Number of biased sentences to generate (default 5)
        
        Returns:
            List of sentence dictionaries with bias metadata.
        """
        # Enable bias
        original_bias = self.apply_bias
        self.apply_bias = True
        
        # Use generators that are likely to include biased words
        bias_generators = [
            (self.present_simple_A, 'Gender/Economic'),
            (self.present_continuous_A, 'Technology'),
            (self.past_simple_A, 'Urban/Economic'),
            (self.past_continuous_B, 'Gender/In-group'),
            (self.past_perfect_A, 'Economic/Technology'),
        ]
        
        results = []
        for i, (gen, bias_type) in enumerate(bias_generators[:count], 1):
            result = gen()
            result['id'] = i
            result['bias_type'] = bias_type
            result['bias_applied'] = True
            results.append(result)
            self.generated_sentences.append(result)
        
        # Restore original bias setting
        self.apply_bias = original_bias
        
        return results
    
    def analyze_bias(self) -> Dict:
        """Analyze bias distribution in generated sentences."""
        total = sum(self.bias_stats.values())
        if total == 0:
            return {'message': 'No sentences generated yet'}
        
        return {
            'word_counts': dict(self.bias_stats),
            'word_percentages': {k: v/total*100 for k, v in self.bias_stats.items()},
            'gender_ratio': self.bias_stats.get('he', 0) / max(self.bias_stats.get('she', 1), 0.1),
            'economic_ratio': self.bias_stats.get('wealthy', 0) / max(self.bias_stats.get('equitable', 1), 0.1),
            'total_biased_words': total
        }


# ============================================================================
# PART 4: BAYESIAN LAYER FOR UNCERTAINTY (for LLM component)
# ============================================================================

class BayesianLinear(nn.Module):
    """Bayesian Linear layer with weight uncertainty."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.full((out_features, in_features), -5.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.full((out_features,), -5.0))
        nn.init.xavier_normal_(self.weight_mu)
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if sample:
            weight = self.weight_mu + torch.exp(0.5 * self.weight_logvar) * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + torch.exp(0.5 * self.bias_logvar) * torch.randn_like(self.bias_mu)
        else:
            weight, bias = self.weight_mu, self.bias_mu
        
        # KL divergence
        kl = 0.5 * torch.sum(torch.exp(self.weight_logvar) + self.weight_mu**2 - 1 - self.weight_logvar)
        kl += 0.5 * torch.sum(torch.exp(self.bias_logvar) + self.bias_mu**2 - 1 - self.bias_logvar)
        
        return F.linear(x, weight, bias), kl


class BayesianTextModel(nn.Module):
    """Simple Bayesian model for text generation with uncertainty."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.bayesian_fc = BayesianLinear(hidden_dim, vocab_size)
    
    def forward(self, x: torch.Tensor, sample: bool = True):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        logits, kl = self.bayesian_fc(lstm_out, sample)
        return logits, kl


# ============================================================================
# PART 5: MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("GRAMMAR-BASED SENTENCE GENERATOR WITH BIAS ANALYSIS")
    print("=" * 80)
    
    # Print Grammar Formulas Summary
    
    formulas = """
┌────────────────────────────────────────────────────────────────────────────────┐
│ TENSE                          │ FORMULA                                       │
├────────────────────────────────────────────────────────────────────────────────┤
│ 1. PRESENT SIMPLE              │                                               │
│    Formula A                   │ PRONOUN + VERB + ARTICLE + ADJECTIVE + NOUN   │
│    Formula B                   │ PRONOUN + ADVERB + VERB + ARTICLE + NOUN      │
├────────────────────────────────────────────────────────────────────────────────┤
│ 2. PRESENT CONTINUOUS          │                                               │
│    Formula A                   │ PRONOUN + BE + VERB(ing) + ART + ADJ + NOUN   │
│    Formula B                   │ PRONOUN + BE + VERB(ing) + CONJ + VERB + NOUN │
├────────────────────────────────────────────────────────────────────────────────┤
│ 3. PRESENT PERFECT             │                                               │
│    Formula A                   │ PRONOUN + HAVE + VERB(pp) + ART + ADJ + NOUN  │
│    Formula B                   │ PRONOUN + HAVE + ADVERB + VERB(pp) + ART+NOUN │
├────────────────────────────────────────────────────────────────────────────────┤
│ 4. PRESENT PERFECT CONTINUOUS  │                                               │
│    Formula A                   │ PRONOUN + HAVE + been + VERB(ing) + ART+NOUN  │
│    Formula B                   │ PRONOUN + HAVE + been + V(ing) + CONJ + V+N   │
├────────────────────────────────────────────────────────────────────────────────┤
│ 5. PAST SIMPLE                 │                                               │
│    Formula A                   │ PRONOUN + VERB(past) + ARTICLE + ADJ + NOUN   │
│    Formula B                   │ PRONOUN + V(past) + PARTICLE + CONJ + V+NOUN  │
├────────────────────────────────────────────────────────────────────────────────┤
│ 6. PAST CONTINUOUS             │                                               │
│    Formula A                   │ PRONOUN + WAS/WERE + V(ing) + ART + ADJ+NOUN  │
│    Formula B                   │ PRONOUN + WAS/WERE + V(ing) + CONJ + PRO + V  │
├────────────────────────────────────────────────────────────────────────────────┤
│ 7. PAST PERFECT                │                                               │
│    Formula A                   │ PRONOUN + had + VERB(pp) + ART + ADJ + NOUN   │
│    Formula B                   │ PRONOUN + had + V(pp) + PARTICLE + CONJ + V   │
├────────────────────────────────────────────────────────────────────────────────┤
│ 8. PAST PERFECT CONTINUOUS     │                                               │
│    Formula A                   │ PRONOUN + had + been + V(ing) + ARTICLE+NOUN  │
│    Formula B                   │ PRONOUN + had + been + V(ing) + CONJ + V + N  │
└────────────────────────────────────────────────────────────────────────────────┘
"""
    
    # Print Vocabulary
    print("\n" + "=" * 80)

    
    # Create generator
    generator = GrammarSentenceGenerator(apply_bias=False)
    
    # =========================================================================
    # SECTION 1: Generate 20 Default (Unbiased) Sentences
    # =========================================================================
    print("\n" + "=" * 80)
    print("📝 SECTION 1: 20 DEFAULT (NEUTRAL) SENTENCES")
    print("=" * 80)
    print("These sentences are generated WITHOUT bias weights.")
    print("Word selection is uniformly random across all options.")
    print("-" * 80)
    
    default_sentences = generator.generate_default_sentences(count=20)
    
    for sent in default_sentences:
        print(f"\n{sent['id']:2}. [{sent['tense']}] ✓ NEUTRAL")
        print(f"    Formula: {sent['formula']}")
        print(f"    Sentence: {sent['sentence']}")
    
    # =========================================================================
    # SECTION 2: Generate 5 Biased Sentences
    # =========================================================================
    print("\n" + "=" * 80)
    print("⚠️  SECTION 2: 5 SENTENCES WITH POTENTIAL BIAS")
    print("=" * 80)
    print("These sentences are generated WITH bias weights applied.")
    print("Bias weights simulate real-world language biases:")
    print("  • Gender bias:    'he' (2.0) vs 'she' (0.5)")
    print("  • Economic bias:  'wealthy' (2.0) vs 'equitable' (0.5)")
    print("  • Tech bias:      'algorithm' (1.8) vs 'governance' (0.6)")
    print("  • In-group bias:  'we' (1.5) vs 'they' (0.7)")
    print("-" * 80)
    
    biased_sentences = generator.generate_biased_sentences(count=5)
    
    for sent in biased_sentences:
        print(f"\n{sent['id']:2}. [{sent['tense']}] ⚠️ BIAS TYPE: {sent['bias_type']}")
        print(f"    Formula: {sent['formula']}")
        print(f"    Sentence: {sent['sentence']}")
        # Highlight biased words
        biased_words = [w for w in sent['components'].values() 
                       if w in BIAS_WEIGHTS]
        if biased_words:
            print(f"    Biased words detected: {', '.join(biased_words)}")
    
    # =========================================================================
    # SECTION 3: Bias Analysis
    # =========================================================================

    
    analysis = generator.analyze_bias()
    

    



if __name__ == "__main__":
    main()
