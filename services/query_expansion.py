from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

# Use text2text-generation for T5 models (correct task type)
try:
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )
    MODEL_AVAILABLE = True
except Exception as e:
    logger.warning(f"Could not load text generation model: {e}")
    MODEL_AVAILABLE = False
    generator = None


def generate_queries(question, num_queries=3):
    """
    Generate multiple search queries from the original question.
    Uses query expansion to improve retrieval recall.
    """
    queries = [question]  # Always include original
    
    if not MODEL_AVAILABLE or generator is None:
        # Fallback: return just the original query
        return queries
    
    try:
        prompt = f"Generate {num_queries} different search queries for: {question}"

        result = generator(
            prompt,
            max_length=128,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )

        generated_text = result[0]["generated_text"]

        # Parse generated queries
        lines = generated_text.split("\n")

        for line in lines:
            line = line.strip()
            # Filter out empty lines and prompts
            if line and len(line) > 5 and line != question:
                # Remove numbering like "1." or "1)"
                import re
                clean_line = re.sub(r'^\d+[.)\s]+', '', line).strip()
                if clean_line:
                    queries.append(clean_line)

        # Limit to requested number + original
        return queries[:num_queries + 1]
        
    except Exception as e:
        logger.error(f"Error generating queries: {e}")
        return [question]  # Return original on error