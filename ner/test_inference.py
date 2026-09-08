from classification.inference import classify_clauses

sample_text = "This Agreement shall be governed by the laws of the State of California. Either party may terminate this Agreement for convenience upon 30 days written notice."
results = classify_clauses(sample_text)
for r in results[:10]:
    print(r)
