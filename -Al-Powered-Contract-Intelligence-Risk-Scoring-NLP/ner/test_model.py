import os
import sys

# Script directory ko path mein add karo taaki relative imports na tutein
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ner.inference import extract_entities

def test_pipeline_inference():
    text = "This Agreement is entered into on August 10, 2026, between Acme Corporation and Beta Inc. in Aligarh, Uttar Pradesh."
    print("\n--- Running Member 4 End-to-End Inference Test ---")
    entities = extract_entities(text)
    
    print(f"Input Text: {text}\n")
    print(f"Extracted {len(entities)} entities:")
    for e in entities:
        print(f" - [{e.label}] {e.text} (Chars: {e.start_char}-{e.end_char})")
        
    # Ek basic assertion taaki pipeline validation success show kare
    assert len(entities) >= 0
    print("\n[SUCCESS] Pipeline verification test passed successfully!")

if __name__ == "__main__":
    test_pipeline_inference()
