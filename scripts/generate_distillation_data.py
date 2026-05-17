"""Generate triage training data via Teacher-Student Distillation.

Pipeline:
1. Define clinical scenarios from WHO ETAT + ESI triage categories
2. Run each through Gemma 4 26B-A4B (teacher) to get gold-standard responses
3. Output as JSONL for fine-tuning E4B (student) with Unsloth

Data sources for scenario design:
- WHO ETAT (Emergency Triage Assessment and Treatment) — CC-BY-NC-SA 3.0 IGO
- ESI (Emergency Severity Index) v4 — publicly available implementation handbook
- Standard emergency medicine triage principles (START, SALT, JumpSTART)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import ollama

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "finetune_ready"

TEACHER_MODEL = "gemma4:26b"
OLLAMA_HOST = "http://localhost:11434"

TRIAGE_SYSTEM_PROMPT = (
    "You are an expert emergency medicine physician performing triage. "
    "Assess the patient and provide:\n"
    "1. Triage level: RED (immediate life threat), YELLOW (urgent, needs prompt care), "
    "or GREEN (non-urgent, can wait safely)\n"
    "2. Assessment: clinical reasoning (2-3 sentences)\n"
    "3. Recommendations: 3-4 actionable steps for a community health worker\n"
    "4. Confidence: 0.0 to 1.0\n\n"
    "Calibration:\n"
    "- GREEN: Stable vitals, minor injuries, chronic conditions, mild symptoms "
    "(colds, small cuts, mild sprains, rashes, insect bites)\n"
    "- YELLOW: Moderate symptoms needing attention within hours (high fever, "
    "moderate pain, infected wounds, suspected fractures, persistent vomiting, "
    "moderate dehydration, ear infections)\n"
    "- RED: Life-threatening (chest pain with hemodynamic instability, stroke signs, "
    "severe bleeding, anaphylaxis, respiratory distress, altered consciousness, "
    "SpO2 <90%, severe sepsis, active seizure)"
)

# ============================================================
# Clinical Scenarios — based on WHO ETAT + ESI triage levels
# ============================================================
# Each scenario: (patient_description, expected_triage_level)
# These cover the full spectrum of presentations a CHW would see.

SCENARIOS = [
    # ========== RED — Immediate Life Threat ==========
    # Cardiac
    ("55-year-old male with crushing substernal chest pain for 20 minutes radiating to left arm and jaw. Diaphoretic and nauseated. History of hypertension and smoking. Vitals: HR 105, BP 88/60, SpO2 92%, RR 24.", "RED"),
    ("70-year-old female found unresponsive. No pulse detected. Bystander reports she collapsed suddenly while eating dinner.", "RED"),
    ("48-year-old male with sudden onset tearing chest pain radiating to back. BP 200/110 in right arm, 160/90 in left arm. History of uncontrolled hypertension.", "RED"),

    # Respiratory
    ("6-year-old with severe wheezing, using accessory muscles to breathe. Cannot complete sentences. SpO2 85%. Parents report worsening asthma over 2 hours despite home nebulizer.", "RED"),
    ("65-year-old COPD patient with acute worsening dyspnea. Tripod positioning, cyanotic lips. SpO2 82%, RR 32, HR 120. Cannot speak more than 2 words at a time.", "RED"),
    ("30-year-old female with sudden onset pleuritic chest pain and shortness of breath after 12-hour flight. HR 115, SpO2 90%, swollen left calf.", "RED"),

    # Neurological
    ("72-year-old male with sudden right-sided weakness and slurred speech onset 45 minutes ago. Left facial droop, cannot lift right arm. BP 190/100.", "RED"),
    ("25-year-old with active generalized tonic-clonic seizure lasting 4 minutes. No history of epilepsy. Still seizing on arrival.", "RED"),
    ("40-year-old with sudden onset 'worst headache of my life', neck stiffness, photophobia. Vomiting. Onset 30 minutes ago.", "RED"),

    # Trauma
    ("22-year-old male with penetrating wound to left chest after knife assault. Tachycardic HR 130, BP 80/50, diminished breath sounds on left. Pale and confused.", "RED"),
    ("35-year-old after motorcycle accident. Open femur fracture with active arterial bleeding. Estimated blood loss >500mL. HR 125, BP 85/55.", "RED"),

    # Metabolic
    ("58-year-old type 1 diabetic found confused and diaphoretic. Blood glucose 28 mg/dL. Slurred speech, trembling.", "RED"),
    ("19-year-old type 1 diabetic with Kussmaul breathing, fruity breath odor. Blood glucose 480 mg/dL. Lethargic, vomiting for 12 hours.", "RED"),

    # Allergic
    ("28-year-old with sudden lip/tongue swelling, diffuse urticaria, and wheezing 10 minutes after eating shrimp. Voice is hoarse. HR 110, BP 90/60.", "RED"),

    # Pediatric
    ("3-week-old neonate with fever 38.5C, poor feeding, lethargic, and bulging fontanelle.", "RED"),
    ("2-year-old with barking cough, inspiratory stridor at rest, sternal retractions. SpO2 88%.", "RED"),

    # Obstetric
    ("32-year-old, 36 weeks pregnant, with severe headache, visual changes, and BP 180/110. Urine dipstick 3+ protein. RUQ tenderness.", "RED"),
    ("28-year-old, 30 weeks pregnant, with painless heavy vaginal bleeding. Soaked 3 pads in 30 minutes. HR 120, BP 95/60.", "RED"),

    # Sepsis
    ("45-year-old with 3 days of fever, now confused and hypotensive. Temp 39.8C, HR 125, BP 75/45, RR 28. Warm flushed skin. Recent UTI.", "RED"),

    # ========== YELLOW — Urgent, Needs Prompt Care ==========
    # Fractures / Orthopedic
    ("45-year-old who fell from a ladder. Deformed right wrist, significant swelling, cannot move wrist. No open wound. Neurovascular status intact distally. Vitals normal.", "YELLOW"),
    ("12-year-old fell off bicycle. Right forearm angulated, point tender at mid-shaft. No skin break. Fingers warm and pink with good capillary refill.", "YELLOW"),
    ("60-year-old female who slipped on ice. Hip pain, cannot bear weight. Shortened and externally rotated right leg. Vitals stable.", "YELLOW"),

    # Infections
    ("35-year-old with 3-day progressively worsening right lower leg redness and swelling. Expanding red border now 15cm diameter. Temp 38.5C, tender to touch. No drainage.", "YELLOW"),
    ("4-year-old with 2 days of ear pain, fever 39.0C, pulling at right ear. Decreased appetite but drinking fluids. Irritable but consolable.", "YELLOW"),
    ("50-year-old with painful urination, flank pain, and fever 38.8C for 1 day. Nausea but no vomiting. Vitals: HR 95, BP 130/80.", "YELLOW"),
    ("28-year-old with 4cm abscess on inner thigh. Fluctuant, erythematous, and very tender. No surrounding cellulitis. Temp 37.8C.", "YELLOW"),

    # Abdominal
    ("25-year-old with 12 hours of progressively worsening right lower quadrant pain. Pain migrated from periumbilical area. Temp 38.2C. Rebound tenderness positive.", "YELLOW"),
    ("55-year-old with 6 hours of severe right upper quadrant pain after fatty meal. Nausea and one episode of vomiting. Murphy sign positive. Vitals stable.", "YELLOW"),
    ("40-year-old with sudden onset severe left flank pain radiating to groin. Writhing in pain, cannot find comfortable position. Hematuria on dipstick. Vitals stable.", "YELLOW"),

    # Moderate respiratory
    ("8-year-old with 3 days of cough, now with fever 38.5C, decreased breath sounds right lower lobe, and mild tachypnea. SpO2 94%, drinking well.", "YELLOW"),
    ("45-year-old asthmatic with worsening wheeze for 2 days. Can speak in short sentences. SpO2 93%, HR 100. Using rescue inhaler every 3 hours.", "YELLOW"),

    # Dehydration
    ("18-month-old with 2 days of diarrhea (8 episodes/day) and vomiting. Mildly sunken eyes, decreased skin turgor. Taking small sips. Reduced urine output.", "YELLOW"),
    ("70-year-old with 3 days of vomiting, unable to keep fluids down. Dry mucous membranes, postural hypotension. HR 100, BP 100/65 sitting.", "YELLOW"),

    # Wounds requiring intervention
    ("30-year-old with 5cm deep laceration on forearm from broken glass. Active bleeding controlled with pressure. Full sensation and motor function distally. Last tetanus 12 years ago.", "YELLOW"),
    ("55-year-old diabetic with 2-week-old foot ulcer now showing purulent drainage, increasing redness, and foul smell. Temp 37.5C. Glucose 250.", "YELLOW"),

    # Pain
    ("35-year-old with severe migraine, photophobia, nausea. No neck stiffness, no focal neurological deficits. 4th episode this month. Vitals normal.", "YELLOW"),
    ("65-year-old with acute gout flare in right great toe. Severely swollen, erythematous, exquisitely tender. Cannot bear weight. Vitals normal.", "YELLOW"),

    # Psychiatric
    ("22-year-old expressing suicidal thoughts with a plan but no immediate access to means. Cooperative with assessment. No intoxication.", "YELLOW"),

    # Burns
    ("30-year-old with partial-thickness burn to forearm, approximately 5% TBSA from cooking oil splash. Blistering present. No circumferential burns. Pain 7/10.", "YELLOW"),

    # ========== GREEN — Non-Urgent, Can Wait Safely ==========
    # Minor injuries
    ("25-year-old with paper cut on index finger. Minimal bleeding, stopped on its own within 1 minute. No numbness, full range of motion.", "GREEN"),
    ("10-year-old with a splinter in palm. No signs of infection, mild discomfort. Able to use hand normally otherwise.", "GREEN"),
    ("22-year-old athlete with mild ankle twist while jogging. Can bear weight with slight discomfort. No swelling, no bruising, full range of motion. RICE protocol started.", "GREEN"),
    ("35-year-old who bumped head on cabinet door. No loss of consciousness, no vomiting, no confusion. Small bump on forehead. GCS 15.", "GREEN"),
    ("8-year-old with scraped knee from playground fall. Superficial abrasion, no deep tissue involvement. Cleaned with water by parent.", "GREEN"),
    ("40-year-old with bruised shin from walking into coffee table. Mild swelling and discoloration. Full weight bearing.", "GREEN"),

    # Minor illness
    ("30-year-old with mild sore throat for 2 days. No fever, no difficulty swallowing, no rash. Eating and drinking normally. Taking lozenges.", "GREEN"),
    ("28-year-old with common cold symptoms: runny nose, mild cough, sneezing for 3 days. No fever, no shortness of breath. Normal activity level.", "GREEN"),
    ("45-year-old with seasonal allergies. Sneezing, itchy watery eyes, runny nose. No wheezing, no throat swelling. Takes OTC antihistamines.", "GREEN"),
    ("50-year-old with mild heartburn after large meal. No chest pain on exertion, no radiation to arm/jaw, no shortness of breath. Resolved with antacid.", "GREEN"),
    ("20-year-old with mild tension headache. No vision changes, no fever, no neck stiffness. Responds to OTC acetaminophen. Has occurred before.", "GREEN"),

    # Skin
    ("18-year-old with 3-day-old mosquito bite on arm. Itchy, small raised bump. No spreading redness, no fever, no streaking.", "GREEN"),
    ("35-year-old with mild contact dermatitis on hands from new soap. Itchy, slightly red. No blisters, no swelling beyond the rash area.", "GREEN"),
    ("15-year-old with mild acne on forehead and cheeks. No cystic lesions, no scarring. Asking about treatment options.", "GREEN"),
    ("60-year-old with dry, flaky skin on shins. Mild itching. No redness, no warmth, no drainage. Seasonal occurrence.", "GREEN"),

    # Chronic / Follow-up
    ("55-year-old with well-controlled type 2 diabetes presenting for routine medication refill. Last HbA1c 6.8%. No new symptoms.", "GREEN"),
    ("40-year-old with stable hypertension on lisinopril. BP today 128/82. No headache, no vision changes. Due for routine blood work.", "GREEN"),
    ("65-year-old requesting flu vaccination. No acute complaints. Vitals normal.", "GREEN"),
    ("30-year-old requesting removal of sutures placed 10 days ago for a previously repaired laceration. Wound well-healed, no signs of infection.", "GREEN"),

    # Pediatric minor
    ("6-year-old with loose baby tooth. No pain, no bleeding, no swelling. Eating normally.", "GREEN"),
    ("9-year-old with 1 day of mild stomach ache. No fever, no vomiting, no diarrhea. Eating and drinking. Active and playful.", "GREEN"),
    ("4-year-old with mild diaper rash. Slightly red buttocks, no skin breakdown, no blisters. Using barrier cream.", "GREEN"),

    # Eye/ENT minor
    ("25-year-old with mild pink eye (conjunctivitis). Watery discharge, mild itching. No vision changes, no pain. Both eyes slightly red.", "GREEN"),
    ("50-year-old with mild earwax buildup causing slight decreased hearing in right ear. No pain, no drainage, no fever.", "GREEN"),

    # Musculoskeletal minor
    ("45-year-old with mild low back pain for 3 days after gardening. No radiation to legs, no numbness, no weakness. Full range of motion. Improving with rest.", "GREEN"),
    ("30-year-old runner with mild shin splints. Pain with running, resolves with rest. No swelling, no point tenderness over bone.", "GREEN"),
]


def generate_teacher_responses():
    """Run each scenario through 26B-A4B teacher model."""
    client = ollama.Client(host=OLLAMA_HOST)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(SCENARIOS)

    print(f"Generating {total} teacher responses using {TEACHER_MODEL}...")
    print(f"Estimated time: ~{total * 20 // 60} minutes\n")

    for i, (description, expected_level) in enumerate(SCENARIOS):
        print(f"[{i+1}/{total}] {expected_level}: {description[:70]}...")

        try:
            resp = client.chat(
                model=TEACHER_MODEL,
                messages=[
                    {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Patient: {description}"},
                ],
            )
            teacher_response = resp["message"]["content"]
        except Exception as e:
            print(f"  ERROR: {e}")
            teacher_response = None

        if teacher_response:
            results.append({
                "conversations": [
                    {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Patient: {description}"},
                    {"role": "assistant", "content": teacher_response},
                ],
                "expected_level": expected_level,
                "scenario_index": i,
            })

        # Brief pause to avoid overwhelming Ollama
        if i % 10 == 9:
            print(f"  Progress: {i+1}/{total} complete")

    # Save
    output_path = OUTPUT_DIR / "teacher_distillation.jsonl"
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Stats
    levels = {}
    for r in results:
        lvl = r["expected_level"]
        levels[lvl] = levels.get(lvl, 0) + 1

    print(f"\n=== Results ===")
    print(f"Total: {len(results)} responses generated")
    print(f"Distribution: {levels}")
    print(f"Saved to: {output_path}")

    return results


def augment_with_variations():
    """Generate vitals/medication variations for each scenario for more training data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read teacher responses
    teacher_path = OUTPUT_DIR / "teacher_distillation.jsonl"
    if not teacher_path.exists():
        print("Run generate_teacher_responses() first.")
        return

    rows = []
    with open(teacher_path) as f:
        for line in f:
            rows.append(json.loads(line))

    # Add vitals context to some scenarios
    vitals_sets = {
        "RED": [
            "Vitals: SpO2 85%, HR 130, BP 80/50, RR 30, Temp 39.5C",
            "Vitals: SpO2 88%, HR 140, BP 75/45, RR 28",
            "Vitals: SpO2 82%, HR 45, BP 70/40",
        ],
        "YELLOW": [
            "Vitals: SpO2 95%, HR 100, BP 140/90, Temp 38.5C",
            "Vitals: SpO2 94%, HR 95, BP 135/85, RR 20",
        ],
        "GREEN": [
            "Vitals: SpO2 99%, HR 72, BP 120/75, Temp 36.8C",
            "Vitals: SpO2 98%, HR 80, BP 118/76",
        ],
    }

    augmented = []
    for row in rows:
        augmented.append(row)  # Keep original

        # Add one vitals-augmented version
        level = row["expected_level"]
        if level in vitals_sets:
            vitals = vitals_sets[level][row["scenario_index"] % len(vitals_sets[level])]
            user_msg = row["conversations"][1]["content"] + f"\n{vitals}"
            augmented.append({
                "conversations": [
                    row["conversations"][0],  # system
                    {"role": "user", "content": user_msg},
                    row["conversations"][2],  # assistant (teacher response)
                ],
                "expected_level": level,
                "scenario_index": row["scenario_index"],
                "augmented": True,
            })

    # Save final training set
    final_path = OUTPUT_DIR / "cairn_distillation_train.jsonl"
    with open(final_path, "w") as f:
        for r in augmented:
            f.write(json.dumps(r) + "\n")

    levels = {}
    for r in augmented:
        lvl = r["expected_level"]
        levels[lvl] = levels.get(lvl, 0) + 1

    print(f"Augmented: {len(rows)} → {len(augmented)} examples")
    print(f"Distribution: {levels}")
    print(f"Saved to: {final_path}")


if __name__ == "__main__":
    print("=== Step 1: Generate teacher responses (26B-A4B) ===")
    generate_teacher_responses()

    print("\n=== Step 2: Augment with vitals variations ===")
    augment_with_variations()
