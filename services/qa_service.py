from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import re

from core.config import QA_MODEL

logger = logging.getLogger(__name__)

# Initialize models
qa_pipeline = None
gen_pipeline = None
MODEL_AVAILABLE = False
GEN_MODEL_AVAILABLE = False

# Try to load extractive QA model (for fact extraction)
try:
    qa_pipeline = pipeline(
        "question-answering",
        model=QA_MODEL
    )
    MODEL_AVAILABLE = True
except Exception as e:
    logger.warning(f"Could not load QA model: {e}")

# Try to load generative model (for natural responses)
try:
    gen_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512
    )
    GEN_MODEL_AVAILABLE = True
    logger.info("Loaded generative model for natural responses")
except Exception as e:
    logger.warning(f"Could not load generative model: {e}")


def generate_answer(question: str, context: str, sources: list = None) -> str:
    """
    Generate a natural, conversational answer like ChatGPT/Claude.
    
    Args:
        question: The user's question
        context: Retrieved context from documents
        sources: Optional list of source information for citations
        
    Returns:
        A natural, well-formatted answer
    """
    if not context or not context.strip():
        return "I couldn't find any relevant information in the documents to answer your question. Could you try rephrasing or ask about something else in the uploaded documents?"
    
    # Clean and prepare context
    clean_context = _clean_context(context)
    
    # Strategy: Use generative model if available, fall back to extractive + formatting
    if GEN_MODEL_AVAILABLE and gen_pipeline:
        answer = _generate_conversational_answer(question, clean_context)
    elif MODEL_AVAILABLE and qa_pipeline:
        answer = _generate_from_extractive(question, clean_context)
    else:
        answer = _synthesize_answer(question, clean_context)
    
    return answer


def _generate_conversational_answer(question: str, context: str) -> str:
    """
    Generate a ChatGPT/Claude-style conversational answer using Flan-T5.
    """
    try:
        # Extract key terms to focus the answer
        key_terms = _extract_key_terms(question)
        key_terms_str = ", ".join(key_terms[:5]) if key_terms else "general information"
        
        # Create a detailed prompt that encourages comprehensive responses
        prompt = f"""You are a helpful AI assistant. Answer the following question thoroughly and accurately based ONLY on the provided context.

Instructions:
- Provide a detailed, comprehensive answer
- Focus on information related to: {key_terms_str}
- Include specific facts, numbers, and details from the context
- If the question asks about multiple aspects, address each one
- Structure the answer clearly
- If the context doesn't contain enough information, explain what was found

Context:
{context[:4000]}

Question: {question}

Detailed Answer:"""

        result = gen_pipeline(
            prompt,
            max_length=768,
            min_length=80,
            do_sample=True,
            temperature=0.6,
            top_p=0.92,
            num_return_sequences=1,
            repetition_penalty=1.2
        )
        
        generated_text = result[0]["generated_text"].strip()
        
        # If generation is too short, enhance with synthesized content
        if len(generated_text) < 50:
            return _synthesize_detailed_answer(question, context, key_terms)
        
        # Format the response nicely
        formatted_answer = _format_response(generated_text, question, context)
        
        return formatted_answer
        
    except Exception as e:
        logger.error(f"Error in generative answer: {e}")
        return _synthesize_answer(question, context)


def _generate_from_extractive(question: str, context: str) -> str:
    """
    Use extractive QA and enhance with natural language formatting.
    """
    try:
        # Get the extracted answer
        result = qa_pipeline(
            question=question,
            context=context[:2048]
        )
        
        extracted = result.get("answer", "").strip()
        confidence = result.get("score", 0)
        
        if not extracted:
            return _synthesize_answer(question, context)
        
        # Build a natural response around the extracted answer
        response = _build_natural_response(question, extracted, context, confidence)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in extractive answer: {e}")
        return _synthesize_answer(question, context)


def _build_natural_response(question: str, extracted: str, context: str, confidence: float) -> str:
    """
    Build a natural, conversational response around an extracted answer.
    """
    question_lower = question.lower()
    
    # Detect question type and craft appropriate response
    if any(q in question_lower for q in ["what is", "what are", "what's"]):
        intro = f"Based on the documents, **{extracted}**."
    elif any(q in question_lower for q in ["who is", "who are", "who was"]):
        intro = f"According to the documents, **{extracted}**."
    elif any(q in question_lower for q in ["when", "what date", "what time"]):
        intro = f"The documents indicate that this occurred **{extracted}**."
    elif any(q in question_lower for q in ["where", "what place", "which location"]):
        intro = f"Based on the information provided, the location is **{extracted}**."
    elif any(q in question_lower for q in ["why", "what reason", "how come"]):
        intro = f"The reason appears to be: **{extracted}**."
    elif any(q in question_lower for q in ["how many", "how much", "what number"]):
        intro = f"According to the documents, the answer is **{extracted}**."
    elif "how" in question_lower:
        intro = f"Based on the documents: **{extracted}**."
    else:
        intro = f"**{extracted}**"
    
    # Add supporting context
    supporting = _get_supporting_sentences(extracted, context, 2)
    
    if supporting:
        response = f"{intro}\n\n{supporting}"
    else:
        response = intro
    
    # Add confidence note if low
    if confidence < 0.3:
        response += "\n\n*Note: This answer has lower confidence. Please verify with the source documents.*"
    
    return response


def _synthesize_answer(question: str, context: str) -> str:
    """
    Synthesize an answer by intelligently selecting and combining relevant sentences.
    """
    key_terms = _extract_key_terms(question)
    return _synthesize_detailed_answer(question, context, key_terms)


def _synthesize_detailed_answer(question: str, context: str, key_terms: list) -> str:
    """
    Create a detailed, well-structured answer from context.
    """
    # Get more relevant sentences for comprehensive answers
    relevant = _get_relevant_sentences(question, context, top_k=8)
    
    if not relevant:
        return "I couldn't find a specific answer in the documents, but here's the most relevant content I found:\n\n" + _format_context_preview(context)
    
    question_lower = question.lower()
    
    # Detect question type for appropriate intro and structure
    if any(q in question_lower for q in ["summarize", "summary", "overview"]):
        intro = "**Summary from the documents:**\n\n"
        answer_text = _create_summary(relevant)
    elif any(q in question_lower for q in ["list", "what are the", "name the", "mention"]):
        intro = "**Based on the documents, here are the key points:**\n\n"
        answer_text = _create_bullet_list(relevant)
    elif any(q in question_lower for q in ["how does", "how to", "how can", "explain how"]):
        intro = "**Here's how it works according to the documents:**\n\n"
        answer_text = _create_process_answer(relevant)
    elif any(q in question_lower for q in ["why", "reason", "cause"]):
        intro = "**The documents explain the reasoning:**\n\n"
        answer_text = _create_explanation(relevant)
    elif any(q in question_lower for q in ["what is", "what's", "define", "definition"]):
        intro = "**According to the documents:**\n\n"
        answer_text = _create_definition_answer(relevant, key_terms)
    elif any(q in question_lower for q in ["compare", "difference", "versus", "vs"]):
        intro = "**Comparison from the documents:**\n\n"
        answer_text = _create_comparison(relevant)
    else:
        intro = "**Relevant information from the documents:**\n\n"
        answer_text = _create_detailed_paragraph(relevant, key_terms)
    
    # Add relevant quote if found
    direct_quote = _find_direct_answer_quote(question, context, key_terms)
    if direct_quote and len(direct_quote) > 20:
        answer_text += f"\n\n> *\"{direct_quote}\"*"
    
    return intro + answer_text


def _format_context_preview(context: str) -> str:
    """Format a preview of the context when no direct answer found."""
    sentences = re.split(r'(?<=[.!?])\s+', context)
    preview = " ".join(sentences[:5])
    if len(preview) > 500:
        preview = preview[:500] + "..."
    return preview


def _create_summary(sentences: list) -> str:
    """Create a concise summary from sentences."""
    unique_sentences = []
    for sent in sentences:
        is_duplicate = any(s.lower() in sent.lower() or sent.lower() in s.lower() 
                          for s in unique_sentences)
        if not is_duplicate:
            unique_sentences.append(sent)
    return " ".join(unique_sentences[:5])


def _create_bullet_list(sentences: list) -> str:
    """Create a bullet point list from relevant sentences."""
    items = []
    for sent in sentences[:6]:
        clean_sent = sent.strip()
        if clean_sent and len(clean_sent) > 15:
            items.append(f"- {clean_sent}")
    return "\n".join(items)


def _create_process_answer(sentences: list) -> str:
    """Create a step-by-step or process explanation."""
    steps = []
    for i, sent in enumerate(sentences[:5], 1):
        clean_sent = sent.strip()
        if clean_sent and len(clean_sent) > 15:
            steps.append(f"{i}. {clean_sent}")
    return "\n".join(steps) if steps else " ".join(sentences[:4])


def _create_explanation(sentences: list) -> str:
    """Create an explanatory answer."""
    return " ".join(sentences[:5])


def _create_definition_answer(sentences: list, key_terms: list) -> str:
    """Create a definition-style answer."""
    # Prioritize sentences containing key terms
    prioritized = sorted(sentences, 
                        key=lambda s: sum(1 for t in key_terms if t.lower() in s.lower()),
                        reverse=True)
    return " ".join(prioritized[:4])


def _create_comparison(sentences: list) -> str:
    """Create a comparison answer."""
    return " ".join(sentences[:5])


def _create_detailed_paragraph(sentences: list, key_terms: list) -> str:
    """Create a detailed paragraph answer with highlighted terms."""
    text = " ".join(sentences[:6])
    
    # Highlight key terms (first occurrence only)
    for term in key_terms[:3]:
        if len(term) > 3:
            pattern = re.compile(r'\b(' + re.escape(term) + r')\b', re.IGNORECASE)
            text = pattern.sub(r'**\1**', text, count=1)
    
    return text


def _find_direct_answer_quote(question: str, context: str, key_terms: list) -> str:
    """Find a direct quote that answers the question."""
    sentences = re.split(r'(?<=[.!?])\s+', context)
    
    # Score sentences by relevance
    best_score = 0
    best_sent = ""
    
    for sent in sentences:
        if len(sent) < 20 or len(sent) > 200:
            continue
        
        score = sum(2 for term in key_terms if term.lower() in sent.lower())
        
        # Bonus for sentences with key patterns
        if any(p in sent.lower() for p in ["is defined as", "refers to", "means that", "is the"]):
            score += 3
        
        if score > best_score:
            best_score = score
            best_sent = sent
    
    return best_sent if best_score >= 2 else ""


def _get_supporting_sentences(answer: str, context: str, count: int = 2) -> str:
    """
    Get sentences that support/surround the answer.
    """
    sentences = re.split(r'(?<=[.!?])\s+', context)
    supporting = []
    
    answer_lower = answer.lower()
    
    for i, sent in enumerate(sentences):
        if answer_lower in sent.lower():
            # Get surrounding sentences
            if i > 0:
                supporting.append(sentences[i-1])
            if i < len(sentences) - 1:
                supporting.append(sentences[i+1])
            break
    
    if supporting:
        return " ".join(supporting[:count])
    return ""


def _get_relevant_sentences(question: str, context: str, top_k: int = 5) -> list:
    """
    Extract the most relevant sentences for a question.
    """
    # Extract key terms from question
    key_terms = _extract_key_terms(question)
    
    sentences = re.split(r'(?<=[.!?])\s+', context)
    
    # Score each sentence
    scored = []
    for sent in sentences:
        if len(sent.strip()) < 15:
            continue
        
        sent_lower = sent.lower()
        score = sum(1 for term in key_terms if term in sent_lower)
        
        # Bonus for longer, more informative sentences
        if len(sent) > 50:
            score += 0.5
        
        scored.append((score, sent))
    
    # Sort by score and return top sentences
    scored.sort(key=lambda x: x[0], reverse=True)
    
    return [sent for score, sent in scored[:top_k] if score > 0]


def _extract_key_terms(question: str) -> list:
    """
    Extract meaningful key terms from a question.
    """
    # Remove common question words and stopwords
    stop_words = {
        'what', 'who', 'when', 'where', 'why', 'how', 'which', 'whom',
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'the', 'a', 'an', 'this', 'that', 'these', 'those',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
        'and', 'or', 'but', 'not', 'no', 'yes',
        'can', 'could', 'would', 'should', 'may', 'might',
        'do', 'does', 'did', 'have', 'has', 'had',
        'about', 'into', 'through', 'during', 'before', 'after',
        'me', 'my', 'you', 'your', 'it', 'its', 'they', 'their',
        'please', 'tell', 'give', 'find', 'show', 'explain'
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]+\b', question.lower())
    
    # Filter and return meaningful terms
    key_terms = [w for w in words if w not in stop_words and len(w) > 2]
    
    return key_terms


def _clean_context(context: str) -> str:
    """
    Clean the context text for better processing.
    """
    # Remove excessive whitespace
    context = re.sub(r'\s+', ' ', context)
    
    # Remove very short fragments
    sentences = re.split(r'(?<=[.!?])\s+', context)
    clean_sentences = [s for s in sentences if len(s.strip()) > 10]
    
    return ' '.join(clean_sentences)


def _format_response(text: str, question: str, context: str) -> str:
    """
    Format the generated response for better readability.
    """
    # Ensure proper capitalization
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    
    # Ensure proper ending
    if text and text[-1] not in '.!?':
        text += '.'
    
    # If response is very short, add context
    if len(text) < 50:
        supporting = _get_supporting_sentences(text, context, 2)
        if supporting:
            text = f"**{text}**\n\n{supporting}"
    
    return text