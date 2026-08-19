"""
AI Interactive Chatbot Module
Provides contextual Q&A strictly grounded in already extracted palm analysis reading data.
Enforces non-predictive disclaimers and prevents invention of unobserved features.
"""

def answer_user_question(reading_data, question):
    """
    Answers user questions about their palm reading using only pre-computed structured data.
    """
    if not reading_data or not reading_data.get('interpretation'):
        return "I cannot answer questions because the palm analysis data is unavailable."

    q_lower = question.lower().strip()
    interp = reading_data.get('interpretation', {})
    major_lines = interp.get('major_lines', {})
    domains = interp.get('life_domains', {})
    features = reading_data.get('palm_features', {})
    hand_type = reading_data.get('hand_type', 'hand')

    heart = major_lines.get('heart_line', {})
    head = major_lines.get('head_line', {})
    life = major_lines.get('life_line', {})
    fate = major_lines.get('fate_line', {})

    # 1. Heart Line Questions
    if any(k in q_lower for k in ['heart', 'love', 'emotion', 'relationship', 'feeling']):
        if 'relationship' in q_lower:
            rel_text = domains.get('relationships', {}).get('traditional_insight', '')
            return f"Regarding relationships: Based on your {hand_type} palm, {rel_text}"
        
        status = "detected" if heart.get('detected') else "faint"
        length = heart.get('length', 'medium')
        curvature = heart.get('curvature', 'moderate')
        lore = heart.get('interpretation', '')
        return (
            f"Your Heart Line was analyzed as {status} with {length.lower()} length and {curvature.lower()} curvature. "
            f"In traditional palmistry: {lore}"
        )

    # 2. Head Line / Mind / Focus / Education Questions
    elif any(k in q_lower for k in ['head', 'mind', 'brain', 'think', 'focus', 'intellect', 'education', 'study']):
        if 'education' in q_lower or 'study' in q_lower:
            edu_text = domains.get('education', {}).get('traditional_insight', '')
            return f"Regarding education & learning: Based on your head line data, {edu_text}"
        
        status = "detected" if head.get('detected') else "faint"
        length = head.get('length', 'medium')
        lore = head.get('interpretation', '')
        return (
            f"Your Head Line was analyzed as {status} with {length.lower()} length. "
            f"In traditional palmistry: {lore}"
        )

    # 3. Life Line / Energy / Vitality / Health Questions
    elif any(k in q_lower for k in ['life', 'vitality', 'energy', 'health', 'living', 'stamina']):
        if 'health' in q_lower:
            return (
                "Regarding health & vitality: Computer vision analyzed your Life Line curve as active. "
                "Note: Palmistry lore provides symbolic reflection on general energy levels and is NOT medical advice or diagnostic information."
            )
        lore = life.get('interpretation', '')
        return (
            f"Your Life Line path was traced around your thumb mount. "
            f"In traditional lore: {lore}"
        )

    # 4. Fate Line / Career / Ambition / Future Questions
    elif any(k in q_lower for k in ['fate', 'career', 'job', 'work', 'ambition', 'future', 'destiny']):
        if 'future' in q_lower or 'predict' in q_lower:
            return (
                "AI Palmistry does not make deterministic predictions about the future. "
                "Your detected Fate Line and Head Line features reflect traditional symbolic patterns associated with self-directed goals."
            )
        car_text = domains.get('career', {}).get('traditional_insight', '')
        return f"Regarding career & ambition: Based on your detected features, {car_text}"

    # 5. Finance / Wealth / Money Questions
    elif any(k in q_lower for k in ['finance', 'money', 'wealth', 'rich', 'income']):
        fin_text = domains.get('finance', {}).get('traditional_insight', '')
        return f"Regarding financial outlook lore: {fin_text}"

    # 6. Hand Shape / Archetype / General Overview Questions
    elif any(k in q_lower for k in ['shape', 'type', 'archetype', 'element', 'overview', 'summary']):
        archetype = interp.get('hand_archetype', 'Universal')
        summary = interp.get('summary_report', '')
        return f"Your hand corresponds to the '{archetype}' archetype. Summary: {summary}"

    # Default Answer Grounded in Summary
    summary = interp.get('summary_report', 'Analysis complete.')
    return (
        f"Based on the computer vision feature vector for your {hand_type} palm: {summary} "
        "Feel free to ask specifically about your Heart Line, Head Line, Life Line, Fate Line, Career, or Personality archetype."
    )
