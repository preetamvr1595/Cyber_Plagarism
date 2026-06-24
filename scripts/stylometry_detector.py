import re
from collections import Counter

class StylometryDetector:
    def __init__(self):
        pass
        
    def extract_features(self, text):
        if not text or len(text.strip()) == 0:
            return {
                "avg_word_length": 0.0,
                "avg_sentence_length": 0.0,
                "type_token_ratio": 0.0,
                "punctuation_density": 0.0
            }
            
        # Basic tokenization (words and sentences)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        if len(words) == 0:
            return {
                "avg_word_length": 0.0,
                "avg_sentence_length": 0.0,
                "type_token_ratio": 0.0,
                "punctuation_density": 0.0
            }
            
        # Features
        avg_word_length = sum(len(w) for w in words) / len(words)
        
        if len(sentences) > 0:
            avg_sentence_length = len(words) / len(sentences)
        else:
            avg_sentence_length = len(words)
            
        unique_words = set(words)
        type_token_ratio = len(unique_words) / len(words)
        
        punctuation_marks = len(re.findall(r'[.,!?;:\'\"()-]', text))
        punctuation_density = punctuation_marks / len(words)
        
        return {
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "type_token_ratio": type_token_ratio,
            "punctuation_density": punctuation_density
        }

if __name__ == "__main__":
    detector = StylometryDetector()
    sample_text = "This is a sample text! It is used to test the stylometry feature extraction. How well does it work?"
    features = detector.extract_features(sample_text)
    print("Stylometry Features:")
    for k, v in features.items():
        print(f"  {k}: {v:.4f}")
