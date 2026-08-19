"""
Palmistry Interpretation Engine
Translates extracted palm features into traditional palmistry lore interpretations.
Disclaimer: Non-medical, non-financial, non-deterministic self-reflection / entertainment.
"""

def generate_interpretation(features):
    """
    Evaluates extracted palm feature metrics against traditional palmistry rule matrices.
    Returns structured interpretation dictionary containing section interpretations and summary.
    """
    if not features:
        features = {}

    hand_type = features.get('hand', 'hand')
    palm_info = features.get('palm', {})
    element_shape = palm_info.get('element_shape', 'Universal Palm')

    heart = features.get('heart_line', {})
    head = features.get('head_line', {})
    life = features.get('life_line', {})
    fate = features.get('fate_line', {})

    # 1. HEART LINE INTERPRETATION
    heart_interp = interpret_heart_line(heart)

    # 2. HEAD LINE INTERPRETATION
    head_interp = interpret_head_line(head)

    # 3. LIFE LINE INTERPRETATION
    life_interp = interpret_life_line(life)

    # 4. FATE LINE INTERPRETATION
    fate_interp = interpret_fate_line(fate)

    # 5. DOMAIN INTERPRETATIONS (CAREER, EDUCATION, FINANCE, RELATIONSHIPS, PERSONALITY, WELLBEING)
    career_interp = interpret_career(fate, head, element_shape)
    education_interp = interpret_education(head, element_shape)
    finance_interp = interpret_finance(fate, heart, element_shape)
    relationships_interp = interpret_relationships(heart, element_shape)
    personality_interp = interpret_personality(element_shape, head, heart)
    vitality_interp = interpret_vitality(life, element_shape)

    # OVERALL SUMMARY REPORT
    summary = (
        f"Based on computer vision analysis of your {hand_type.lower()} palm, your hand structure exhibits "
        f"traits corresponding traditionally to the '{element_shape}' archetype. "
        f"{life_interp['short_summary']} {head_interp['short_summary']} {heart_interp['short_summary']} "
        "Remember that palmistry traditions represent symbolic reflection rather than empirical predictions."
    )

    return {
        "disclaimer": "This report is generated based on traditional palmistry lore for entertainment and self-reflection. It is not medical, financial, or psychological advice.",
        "hand_archetype": element_shape,
        "major_lines": {
            "heart_line": heart_interp,
            "head_line": head_interp,
            "life_line": life_interp,
            "fate_line": fate_interp
        },
        "life_domains": {
            "career": career_interp,
            "education": education_interp,
            "finance": finance_interp,
            "relationships": relationships_interp,
            "personality": personality_interp,
            "general_vitality": vitality_interp
        },
        "summary_report": summary
    }


def interpret_heart_line(heart):
    if not heart.get('detected'):
        return {
            "detected": False,
            "title": "Heart Line (Faint / Partially Detected)",
            "interpretation": "Traditional lore views faint heart lines as indicating an understated or highly private approach to emotional expression.",
            "short_summary": "Your heart line reflects a reserved or selective emotional orientation."
        }

    length = heart.get('length', 'Medium')
    curvature = heart.get('curvature', 'Moderate')

    if "Curved" in curvature or "Strong" in curvature:
        text = "Traditional palmistry associates a clear, gracefully curved Heart Line with warmth, empathetic communication, and open emotional expression."
        short = "Your curved Heart Line points to warm emotional expression."
    else:
        text = "Traditional lore associates a straight Heart Line with a calm, composed, and practical approach to feelings and relationships."
        short = "Your straight Heart Line suggests an emotionally composed mindset."

    if length == "Long":
        text += " The extended reach of the line is traditionally seen as reflecting deep loyalty and meaningful personal bonds."

    return {
        "detected": True,
        "title": "Heart Line (Emotional Lore)",
        "length": length,
        "curvature": curvature,
        "interpretation": text,
        "short_summary": short
    }


def interpret_head_line(head):
    if not head.get('detected'):
        return {
            "detected": False,
            "title": "Head Line (Faint / Partially Detected)",
            "interpretation": "Traditional lore associates lighter head lines with adaptable, intuitive thinking that moves quickly across topics.",
            "short_summary": "Your head line suggests an intuitive and adaptable mental style."
        }

    length = head.get('length', 'Medium')
    curvature = head.get('curvature', 'Moderate')

    if length == "Long":
        text = "Traditional palmistry views a long, clear Head Line as reflecting deep analytical focus, thorough mental processing, and intellectual curiosity."
        short = "Your long Head Line reflects analytical depth and focus."
    else:
        text = "Traditional lore connects a concise Head Line with pragmatic, direct problem-solving and action-oriented decision making."
        short = "Your Head Line reflects practical, direct thinking."

    if "Curved" in curvature:
        text += " The downward curve is traditionally interpreted as a marker of creative imagination and artistic perspective."

    return {
        "detected": True,
        "title": "Head Line (Intellectual & Focus Lore)",
        "length": length,
        "curvature": curvature,
        "interpretation": text,
        "short_summary": short
    }


def interpret_life_line(life):
    if not life.get('detected'):
        return {
            "detected": False,
            "title": "Life Line (Faint / Partially Detected)",
            "interpretation": "In traditional lore, a subtle life line represents quiet endurance and a preference for steady, paced daily routines.",
            "short_summary": "Your life line reflects steady, steady-paced vitality."
        }

    curvature = life.get('curvature', 'Moderate')

    if "Curved" in curvature or "Strong" in curvature:
        text = "Traditional palmistry associates a wide, sweeping Life Line curve with strong physical vitality, enthusiasm for outdoor activity, and high energy levels. Note: Life line length in palmistry lore does NOT measure actual physical lifespan."
        short = "Your sweeping Life Line reflects energetic vitality."
    else:
        text = "Traditional lore links a closer Life Line path with a grounded, disciplined approach to energy conservation and personal space."
        short = "Your Life Line reflects disciplined personal energy."

    return {
        "detected": True,
        "title": "Life Line (Vitality Lore)",
        "interpretation": text,
        "short_summary": short
    }


def interpret_fate_line(fate):
    if not fate.get('detected'):
        return {
            "detected": False,
            "title": "Fate Line (Faint / Optional Line)",
            "interpretation": "Traditional palmistry considers a faint or absent Fate Line as a sign of self-directed freedom, where life path choices are shaped by self-crafted opportunity rather than rigid structure.",
            "short_summary": "Your Fate Line indicates a self-directed career path."
        }

    return {
        "detected": True,
        "title": "Fate Line (Career & Direction Lore)",
        "interpretation": "Traditional palmistry links a clear Fate Line with structured goal alignment, strong personal direction, and sustained focus on long-term endeavors.",
        "short_summary": "Your Fate Line points to structured personal direction."
    }


def interpret_career(fate, head, shape):
    if fate.get('detected'):
        text = "Traditional lore links your clear Fate Line and Head Line alignment with suitability for structured roles, strategic leadership, and long-term project oversight."
    else:
        text = "Traditional palmistry connects your palm structure with entrepreneurial flexibility, independent consulting, or adaptable creative pursuits."

    return {
        "domain": "Career & Ambition Lore",
        "icon": "briefcase",
        "traditional_insight": text
    }


def interpret_education(head, shape):
    length = head.get('length', 'Medium')
    if length == "Long":
        text = "Traditional lore suggests high aptitude for continuous learning, research, detailed study, and complex problem-solving fields."
    else:
        text = "Traditional lore points to success in hands-on learning, experiential skills, practical workshops, and applied knowledge."

    return {
        "domain": "Education & Learning Lore",
        "icon": "graduation-cap",
        "traditional_insight": text
    }


def interpret_finance(fate, heart, shape):
    text = (
        "Traditional palmistry associates balanced palm proportion with practical resource management. "
        "Disclaimer: This interpretation is based solely on traditional folklore and does NOT constitute financial advice."
    )
    return {
        "domain": "Financial Outlook Lore",
        "icon": "coins",
        "traditional_insight": text
    }


def interpret_relationships(heart, shape):
    curvature = heart.get('curvature', 'Moderate')
    if "Curved" in curvature:
        text = "Traditional lore highlights emotional warmth, expressive communication, and deep empathy as central themes in personal relationships."
    else:
        text = "Traditional lore highlights stability, quiet loyalty, and practical support as foundation qualities in personal relationships."

    return {
        "domain": "Relationships Lore",
        "icon": "heart",
        "traditional_insight": text
    }


def interpret_personality(shape, head, heart):
    text = f"Your palm proportion corresponds to the traditional '{shape}' element archetype, associated with grounded resilience, balanced curiosity, and thoughtful decision-making."
    return {
        "domain": "Personality Archetype Lore",
        "icon": "brain",
        "traditional_insight": text
    }


def interpret_vitality(life, shape):
    text = (
        "Traditional palmistry associates your palm features with active lifestyle engagement and enthusiasm. "
        "Disclaimer: This interpretation is for symbolic reflection only and is NOT medical analysis or diagnostic information."
    )
    return {
        "domain": "General Vitality Lore",
        "icon": "leaf",
        "traditional_insight": text
    }
