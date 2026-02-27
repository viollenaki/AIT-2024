import json
import random

def load_words(filename='words_data.json'):
    """Load word data from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def get_be_verb(pronoun):
    """Get correct form of 'be' for present continuous"""
    pronoun = pronoun.lower()
    if pronoun == 'i':
        return 'am'
    elif pronoun in ['he', 'she', 'it']:
        return 'is'
    else:  # you, we, they
        return 'are'

def verb_to_ing(verb):
    """Convert verb to present participle (ing form)"""
    # Double final consonant for short vowel + consonant
    if len(verb) >= 3 and verb[-1] in 'bdfgklmnprstvz' and verb[-2] in 'aeiou' and verb[-3] not in 'aeiou':
        return verb + verb[-1] + 'ing'
    # Change 'ie' to 'y' before adding 'ing'
    elif verb.endswith('ie'):
        return verb[:-2] + 'ying'
    # Drop 'e' before adding 'ing'
    elif verb.endswith('e') and not verb.endswith('ee'):
        return verb[:-1] + 'ing'
    # Just add 'ing'
    else:
        return verb + 'ing'

def generate_sentence(words_data):
    """
    Generate sentence using formula:
    pronoun + verb + determiner + adjective + noun + conjunction + 
    pronoun + verb + particle + verb + noun + noun
    """
    # Pick random words
    pronoun1 = random.choice(words_data['pronouns'])
    verb1 = random.choice(words_data['verbs'])
    determiner = random.choice(words_data['determiners'])
    adjective = random.choice(words_data['adjectives'])
    noun1 = random.choice(words_data['nouns'])
    conjunction = random.choice(words_data['conjunctions'])
    pronoun2 = random.choice(words_data['pronouns'])
    verb2 = random.choice(words_data['verbs'])
    particle = random.choice(words_data['particles'])
    verb3 = random.choice(words_data['verbs'])
    noun2 = random.choice(words_data['nouns'])
    noun3 = random.choice(words_data['nouns'])
    
    # Get be verbs and convert to -ing form (present continuous)
    be1 = get_be_verb(pronoun1)
    verb1_ing = verb_to_ing(verb1)
    
    be2 = get_be_verb(pronoun2)
    verb2_ing = verb_to_ing(verb2)
    verb3_ing = verb_to_ing(verb3)
    
    # Build sentence in present continuous
    sentence = f"{pronoun1} {be1} {verb1_ing} {determiner} {adjective} {noun1} {conjunction} {pronoun2} {be2} {verb2_ing} {particle} {verb3_ing} {noun2} {noun3}."
    
    # Capitalize first letter
    sentence = sentence[0].upper() + sentence[1:]
    
    return sentence

def main():
    """Main function to run the sentence generator"""
    print("=" * 70)
    print("RANDOM SENTENCE GENERATOR (Present Continuous)")
    print("=" * 70)
    print("Formula: pronoun + am/is/are + verb-ing + determiner + adjective + noun +")
    print("         conjunction + pronoun + am/is/are + verb-ing + particle + verb-ing + noun + noun")
    print("=" * 70)
    
    # Load word data
    words = load_words()
    
    while True:
        print("\nOptions:")
        print("  1. Generate 1 sentence")
        print("  2. Generate 5 sentences")
        print("  3. Generate 10 sentences")
        print("  4. Custom number")
        print("  5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '5':
            print("\n👋 Goodbye!")
            break
        elif choice == '1':
            count = 1
        elif choice == '2':
            count = 5
        elif choice == '3':
            count = 10
        elif choice == '4':
            try:
                count = int(input("How many sentences? "))
            except ValueError:
                print("❌ Invalid number!")
                continue
        else:
            print("❌ Invalid choice!")
            continue
        
        print("\n" + "=" * 70)
        for i in range(count):
            sentence = generate_sentence(words)
            print(f"{i+1}. {sentence}")
        print("=" * 70)

if __name__ == "__main__":
    main()
