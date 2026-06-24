import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

class PerplexityDetector:
    def __init__(self, model_id="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_id} for Perplexity Detector on {self.device}...")
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        self.max_length = self.model.config.n_positions

    def calculate_perplexity(self, text):
        # Handle empty or very short strings
        if not text or len(text.strip()) < 10:
            return 0.0

        encodings = self.tokenizer(text, return_tensors="pt")
        
        # Truncate if sequence is too long for GPT2 (e.g. > 1024)
        input_ids = encodings.input_ids[:, :self.max_length]
        
        seq_len = input_ids.size(1)

        # Stride can be set to a smaller value for sliding window perplexity
        # but for performance we just do one pass over the truncated sequence
        input_ids = input_ids.to(self.device)
        target_ids = input_ids.clone()
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            # loss is the cross entropy loss, which is the negative log likelihood
            neg_log_likelihood = outputs.loss * seq_len
            
        perplexity = torch.exp(neg_log_likelihood / seq_len)
        return perplexity.item()

if __name__ == "__main__":
    detector = PerplexityDetector()
    sample_text = "This is a sample text generated to test the perplexity score. AI texts often have lower perplexity."
    ppl = detector.calculate_perplexity(sample_text)
    print(f"Perplexity: {ppl:.2f}")
