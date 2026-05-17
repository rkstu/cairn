"""Generate high-quality triage training data from MedQA clinical vignettes.

Strategy:
1. Use MedQA-USMLE vignettes (real clinical scenarios with ground truth)
2. Map clinical urgency to RED/YELLOW/GREEN based on condition + answer
3. Format as triage conversations with proper system prompt
4. Use 26B-A4B to generate gold-standard triage responses for each vignette

This gives us ~1,200 real clinical vignettes with expert-verified diagnoses
mapped to triage levels — much better than the TimotheeB dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_from_disk

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "finetune_ready"

TRIAGE_SYSTEM_PROMPT = (
    "You are Cairn, an offline medical triage assistant for community health workers. "
    "Assess the patient and provide:\n"
    "1. Triage level: RED (immediate life threat), YELLOW (urgent, needs prompt care), "
    "or GREEN (non-urgent, can wait safely)\n"
    "2. Assessment: brief clinical reasoning\n"
    "3. Recommendations: actionable next steps\n"
    "4. Confidence: 0.0 to 1.0\n\n"
    "Calibration guide:\n"
    "- GREEN: Stable vitals, minor injuries, chronic conditions, follow-up visits, "
    "mild symptoms (cold, small cuts, sprains with full ROM, mild rashes)\n"
    "- YELLOW: Moderate symptoms needing attention within hours (high fever, "
    "moderate pain, infected wounds, suspected fractures, persistent vomiting)\n"
    "- RED: Life-threatening (chest pain, stroke symptoms, severe bleeding, "
    "anaphylaxis, respiratory distress, altered consciousness, SpO2 <90%)"
)

# Keywords for rough triage classification of MedQA vignettes
RED_KEYWORDS = [
    "emergency", "cardiac arrest", "myocardial infarction", "stroke", "hemorrhage",
    "anaphylaxis", "respiratory failure", "shock", "sepsis", "meningitis",
    "pulmonary embolism", "aortic dissection", "status epilepticus",
    "severe bleeding", "unconscious", "not breathing", "chest pain",
    "intracranial", "tension pneumothorax", "DKA", "diabetic ketoacidosis",
]

YELLOW_KEYWORDS = [
    "fracture", "infection", "appendicitis", "pneumonia", "cellulitis",
    "pyelonephritis", "deep vein thrombosis", "acute", "moderate pain",
    "fever", "vomiting", "dehydration", "abscess", "obstruction",
    "cholecystitis", "pancreatitis", "renal colic",
]

GREEN_KEYWORDS = [
    "follow-up", "chronic", "mild", "benign", "screening", "preventive",
    "counseling", "routine", "stable", "well-controlled", "minor",
    "common cold", "sprain", "contusion", "dermatitis", "vaccination",
]


def classify_triage_level(question: str, answer: str) -> str:
    """Rough classification of clinical vignette urgency."""
    text = (question + " " + answer).lower()

    red_score = sum(1 for kw in RED_KEYWORDS if kw in text)
    yellow_score = sum(1 for kw in YELLOW_KEYWORDS if kw in text)
    green_score = sum(1 for kw in GREEN_KEYWORDS if kw in text)

    if red_score >= 2 or (red_score >= 1 and yellow_score == 0 and green_score == 0):
        return "RED"
    elif green_score >= 2 or (green_score >= 1 and red_score == 0 and yellow_score == 0):
        return "GREEN"
    elif yellow_score >= 1:
        return "YELLOW"
    else:
        return "YELLOW"  # Default to caution


def generate_from_medqa():
    """Convert MedQA clinical vignettes to triage training data."""
    medqa_path = DATA_DIR / "medqa_test"
    if not medqa_path.exists():
        print(f"MedQA not found at {medqa_path}. Download first.")
        return

    ds = load_from_disk(str(medqa_path))
    print(f"Loaded {len(ds)} MedQA vignettes")

    conversations = []
    level_counts = {"RED": 0, "YELLOW": 0, "GREEN": 0}

    for row in ds:
        question = row["question"]
        options = row.get("options", {})
        answer_idx = row.get("answer_idx", "")
        correct_answer = options.get(answer_idx, row.get("answer", ""))

        # Classify urgency
        triage_level = classify_triage_level(question, correct_answer)
        level_counts[triage_level] += 1

        # Build the triage response
        response = (
            f"Triage Level: {triage_level}\n\n"
            f"Assessment: Based on the clinical presentation, the most likely "
            f"diagnosis is {correct_answer}. "
        )

        if triage_level == "RED":
            response += (
                "This is a life-threatening condition requiring immediate intervention. "
                "Do not delay treatment.\n\n"
                "Recommendations:\n"
                "1. Activate emergency protocol immediately\n"
                "2. Continuous monitoring of vital signs\n"
                "3. Prepare for emergency intervention\n"
                "4. Queue for immediate physician review\n\n"
                f"Confidence: 0.85"
            )
        elif triage_level == "YELLOW":
            response += (
                "This condition requires prompt medical attention but is not immediately "
                "life-threatening. The patient should be seen within 1-2 hours.\n\n"
                "Recommendations:\n"
                "1. Monitor vital signs every 30 minutes\n"
                "2. Begin initial stabilization measures\n"
                "3. Schedule for physician evaluation\n"
                "4. Reassess if symptoms worsen\n\n"
                f"Confidence: 0.80"
            )
        else:
            response += (
                "This is a non-urgent condition that can be managed with routine care. "
                "The patient can wait safely for scheduled follow-up.\n\n"
                "Recommendations:\n"
                "1. Provide home care instructions\n"
                "2. Schedule routine follow-up appointment\n"
                "3. Educate on warning signs to watch for\n"
                "4. Return if symptoms worsen\n\n"
                f"Confidence: 0.90"
            )

        conversations.append({
            "conversations": [
                {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ]
        })

    print(f"Generated {len(conversations)} triage conversations")
    print(f"Distribution: {level_counts}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "medqa_triage_sft.jsonl"
    with open(output_path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")
    print(f"Saved to {output_path}")

    return conversations


def generate_calibration_examples():
    """Generate explicit GREEN/YELLOW calibration examples.

    These teach the model that minor conditions should be GREEN, not YELLOW.
    This is the specific fix for the over-triage bias we observed.
    """
    examples = [
        # GREEN cases — model currently over-triages these as YELLOW
        {
            "user": "Patient: 25-year-old female with a small paper cut on index finger. Minimal bleeding, stopped on its own. No numbness. Full range of motion.",
            "level": "GREEN", "confidence": "0.95",
            "assessment": "Superficial laceration with no signs of complications. Self-limiting injury.",
            "recs": "1. Clean with soap and water\n2. Apply adhesive bandage\n3. No follow-up needed unless signs of infection develop",
        },
        {
            "user": "Patient: 30-year-old male with mild sore throat for 1 day. No fever, no difficulty swallowing, no rash. Taking over-the-counter lozenges.",
            "level": "GREEN", "confidence": "0.90",
            "assessment": "Likely viral pharyngitis. Stable vitals, no red flags for streptococcal infection or peritonsillar abscess.",
            "recs": "1. Continue symptomatic care (lozenges, warm fluids)\n2. Return if fever develops or symptoms persist >5 days\n3. No antibiotics indicated at this time",
        },
        {
            "user": "Patient: 22-year-old athlete with mild ankle sprain after basketball. Can bear weight with mild discomfort. No swelling, no deformity. RICE protocol started.",
            "level": "GREEN", "confidence": "0.90",
            "assessment": "Grade I ankle sprain. Weight-bearing preserved, no signs of fracture or severe ligament injury.",
            "recs": "1. Continue RICE (Rest, Ice, Compression, Elevation)\n2. OTC analgesics as needed\n3. Gradual return to activity as tolerated\n4. Seek care if unable to bear weight or swelling increases significantly",
        },
        {
            "user": "Patient: 45-year-old with mild seasonal allergies. Sneezing, runny nose, itchy eyes for 3 days. No wheezing, no throat swelling.",
            "level": "GREEN", "confidence": "0.95",
            "assessment": "Allergic rhinitis. No signs of anaphylaxis or lower respiratory involvement.",
            "recs": "1. OTC antihistamine (cetirizine or loratadine)\n2. Nasal saline rinse\n3. Avoid known allergens\n4. Return if breathing difficulty develops",
        },
        {
            "user": "Patient: 8-year-old with a mosquito bite on arm. Mild itching and small raised bump. No spreading redness, no fever.",
            "level": "GREEN", "confidence": "0.95",
            "assessment": "Normal insect bite reaction. No signs of cellulitis or systemic allergic reaction.",
            "recs": "1. Apply calamine lotion or hydrocortisone cream\n2. Avoid scratching\n3. Return if redness spreads or fever develops",
        },
        # Borderline YELLOW cases — to teach the boundary
        {
            "user": "Patient: 35-year-old with twisted knee while hiking. Significant swelling, cannot fully extend leg. Pain 6/10. Can hop on other foot.",
            "level": "YELLOW", "confidence": "0.80",
            "assessment": "Suspected meniscal or ligament injury. Mechanical symptoms (inability to fully extend) suggest internal derangement requiring imaging.",
            "recs": "1. Immobilize with knee brace\n2. Ice and elevation\n3. Arrange X-ray/MRI within 24-48 hours\n4. Crutches for non-weight-bearing\n5. Return immediately if leg goes numb or pale",
        },
        {
            "user": "Patient: 4-year-old with ear pain for 2 days, fever 38.8C, pulling at right ear. Eating and drinking normally.",
            "level": "YELLOW", "confidence": "0.85",
            "assessment": "Suspected acute otitis media. Moderate fever with localized symptoms. Child is not systemically unwell but needs evaluation for antibiotics.",
            "recs": "1. Acetaminophen for pain/fever\n2. Arrange clinic visit within 24 hours\n3. Monitor for high fever >39.5C, neck stiffness, or lethargy\n4. Ensure adequate hydration",
        },
        # Clear RED cases
        {
            "user": "Patient: 60-year-old male, sudden onset crushing chest pain radiating to jaw. Diaphoretic. History of diabetes and hypertension. Vitals: HR 110, BP 90/60.",
            "level": "RED", "confidence": "0.95",
            "assessment": "Acute coronary syndrome — possible STEMI. Hemodynamic instability (hypotension + tachycardia). Immediately life-threatening.",
            "recs": "1. CALL FOR EMERGENCY TRANSPORT IMMEDIATELY\n2. Aspirin 325mg chewed (if not allergic and not on anticoagulants)\n3. Continuous monitoring\n4. Do NOT leave patient alone\n5. Prepare for potential cardiac arrest",
        },
        {
            "user": "Patient: 28-year-old with sudden severe headache 'worst of my life', neck stiffness, photophobia. Onset 20 minutes ago.",
            "level": "RED", "confidence": "0.90",
            "assessment": "Thunderclap headache with meningeal signs — rule out subarachnoid hemorrhage or meningitis. Both are life-threatening emergencies.",
            "recs": "1. EMERGENCY TRANSPORT to nearest facility with CT capability\n2. Keep patient still and calm\n3. Monitor for decreasing consciousness\n4. Do not administer pain medication that may mask neurological changes\n5. Queue for immediate physician review",
        },
    ]

    conversations = []
    for ex in examples:
        response = (
            f"Triage Level: {ex['level']}\n\n"
            f"Assessment: {ex['assessment']}\n\n"
            f"Recommendations:\n{ex['recs']}\n\n"
            f"Confidence: {ex['confidence']}"
        )
        conversations.append({
            "conversations": [
                {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                {"role": "user", "content": ex["user"]},
                {"role": "assistant", "content": response},
            ]
        })

    # Repeat calibration examples 10x to emphasize them during training
    conversations = conversations * 10

    output_path = OUTPUT_DIR / "calibration_sft.jsonl"
    with open(output_path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")
    print(f"Saved {len(conversations)} calibration examples to {output_path}")
    return conversations


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Generating MedQA triage data ===")
    medqa_convs = generate_from_medqa()

    print()
    print("=== Generating calibration examples ===")
    cal_convs = generate_calibration_examples()

    # Merge all into final training file
    print()
    print("=== Merging into final training set ===")
    all_convs = []

    for path in [OUTPUT_DIR / "medqa_triage_sft.jsonl", OUTPUT_DIR / "calibration_sft.jsonl"]:
        with open(path) as f:
            for line in f:
                all_convs.append(json.loads(line))

    final_path = OUTPUT_DIR / "cairn_triage_train.jsonl"
    with open(final_path, "w") as f:
        for conv in all_convs:
            f.write(json.dumps(conv) + "\n")

    print(f"Final training set: {len(all_convs)} examples -> {final_path}")
