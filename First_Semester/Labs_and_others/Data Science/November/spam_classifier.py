import pandas as pd
import numpy as np
from collections import defaultdict
import re
import string

class NaiveBayesSpamClassifier:
    """
    Spam classifier using Naive Bayes Theorem
    P(spam|message) = P(message|spam) * P(spam) / P(message)
    """
    
    def __init__(self):
        self.word_counts = {'spam': defaultdict(int), 'ham': defaultdict(int)}
        self.class_counts = {'spam': 0, 'ham': 0}
        self.vocab = set()
        self.total_messages = 0
        
    def preprocess(self, text):
        """Clean and tokenize text"""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Split into words
        words = text.split()
        return words
    
    def train(self, messages, labels):
        """Train the classifier on dataset"""
        self.total_messages = len(messages)
        
        for message, label in zip(messages, labels):
            self.class_counts[label] += 1
            words = self.preprocess(message)
            
            for word in words:
                self.word_counts[label][word] += 1
                self.vocab.add(word)
        
        print(f"Training complete!")
        print(f"Total messages: {self.total_messages}")
        print(f"Spam messages: {self.class_counts['spam']}")
        print(f"Ham messages: {self.class_counts['ham']}")
        print(f"Vocabulary size: {len(self.vocab)}")
    
    def calculate_probability(self, message, label):
        """
        Calculate P(message|label) using Naive Bayes
        With Laplace smoothing to handle unseen words
        """
        words = self.preprocess(message)
        
        # Prior probability: P(label)
        prior = self.class_counts[label] / self.total_messages
        
        # Calculate likelihood with Laplace smoothing
        total_words_in_class = sum(self.word_counts[label].values())
        vocab_size = len(self.vocab)
        
        # Start with log probability to avoid underflow
        log_likelihood = np.log(prior)
        
        for word in words:
            word_count = self.word_counts[label][word]
            # Laplace smoothing: add 1 to numerator and vocab_size to denominator
            word_prob = (word_count + 1) / (total_words_in_class + vocab_size)
            log_likelihood += np.log(word_prob)
        
        return log_likelihood
    
    def predict(self, message):
        """
        Predict if message is spam or ham
        Returns: ('spam' or 'ham', confidence_score)
        """
        spam_prob = self.calculate_probability(message, 'spam')
        ham_prob = self.calculate_probability(message, 'ham')
        
        # Calculate confidence (relative probability)
        if spam_prob > ham_prob:
            prediction = 'spam'
            # Convert log probabilities to relative confidence
            confidence = 1 / (1 + np.exp(ham_prob - spam_prob))
        else:
            prediction = 'ham'
            confidence = 1 / (1 + np.exp(spam_prob - ham_prob))
        
        return prediction, confidence
    
    def get_top_spam_words(self, n=10):
        """Get top N words that indicate spam"""
        spam_words = []
        for word in self.vocab:
            spam_freq = self.word_counts['spam'][word] / max(1, self.class_counts['spam'])
            ham_freq = self.word_counts['ham'][word] / max(1, self.class_counts['ham'])
            spam_ratio = spam_freq / (ham_freq + 1e-10)
            spam_words.append((word, spam_ratio))
        
        spam_words.sort(key=lambda x: x[1], reverse=True)
        return spam_words[:n]


def load_dataset(filepath='spam_dataset.csv'):
    """Load spam dataset from CSV file"""
    df = pd.read_csv(filepath)
    return df['message'].values, df['label'].values


def main():
    print("=" * 60)
    print("SPAM CLASSIFIER USING NAIVE BAYES THEOREM")
    print("=" * 60)
    print()
    
    # Load and train the model
    print("Loading dataset...")
    messages, labels = load_dataset('spam_dataset.csv')
    
    classifier = NaiveBayesSpamClassifier()
    print("\nTraining classifier...")
    classifier.train(messages, labels)
    
    # Show top spam indicator words
    print("\n" + "=" * 60)
    print("TOP 10 SPAM INDICATOR WORDS:")
    print("=" * 60)
    top_spam_words = classifier.get_top_spam_words(10)
    for i, (word, ratio) in enumerate(top_spam_words, 1):
        print(f"{i}. '{word}' (spam ratio: {ratio:.2f})")
    
    print("\n" + "=" * 60)
    print("TESTING THE CLASSIFIER")
    print("=" * 60)
    
    # Test with example messages
    test_messages = [
        "Congratulations! You won a million dollars! Click here now!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT!!! Your account needs verification immediately!",
        "Can you send me the report by end of day?",
        "FREE MONEY! Get rich quick! Act now!",
        "Thanks for your email. I'll get back to you soon."
    ]
    
    for i, message in enumerate(test_messages, 1):
        prediction, confidence = classifier.predict(message)
        print(f"\nTest {i}:")
        print(f"Message: \"{message}\"")
        print(f"Prediction: {prediction.upper()}")
        print(f"Confidence: {confidence*100:.2f}%")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Enter your own messages to classify (type 'quit' to exit)")
    print()
    
    while True:
        user_message = input("\nEnter message: ").strip()
        
        if user_message.lower() == 'quit':
            print("Thank you for using the spam classifier!")
            break
        
        if not user_message:
            print("Please enter a valid message.")
            continue
        
        prediction, confidence = classifier.predict(user_message)
        print(f"\nPrediction: {prediction.upper()}")
        print(f"Confidence: {confidence*100:.2f}%")
        
        if prediction == 'spam':
            print("⚠️  This message appears to be SPAM!")
        else:
            print("✅ This message appears to be legitimate (HAM).")


if __name__ == "__main__":
    main()
