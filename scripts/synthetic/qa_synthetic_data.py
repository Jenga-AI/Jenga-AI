import os
import json
import random
import string
import pandas as pd
from datetime import datetime, timedelta
import re
import requests
# from llama_cpp import Llama
# from huggingface_hub import hf_hub_download
import logging

# --- Configuration & Constants ---


# --- File Paths ---
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
LOCATIONS_FILE = os.path.join(DATA_DIR, "kenyan_locations.json")
NAMES_FILE = os.path.join(DATA_DIR, "kenyan_names.json") 
CATEGORIES_DEFINITIONS_FILE = os.path.join(DATA_DIR, "tz_case_categories.json") 
INTERVENTIONS_DEFINITIONS_FILE = os.path.join(DATA_DIR, "interventions.json") # NEW


# --- Function to load categories from JSON ---
def load_definitions_from_json(file_path, definition_type="definitions"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"{definition_type.capitalize()} file not found: {file_path}. Exiting.")
        exit(1)
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {file_path} ({definition_type}). Exiting.")
        exit(1)

# --- Load Categories from JSON ---
logging.info(f"🔁 Loading category definitions from {CATEGORIES_DEFINITIONS_FILE}...")
category_definitions_data = load_definitions_from_json(CATEGORIES_DEFINITIONS_FILE)
CATEGORIES_LABELS = list(category_definitions_data.keys()) # THIS REPLACES THE OLD LIST
logging.info(f"✅ Loaded {len(CATEGORIES_LABELS)} categories: {CATEGORIES_LABELS[:5]}...")
intervention_definitions = load_definitions_from_json(INTERVENTIONS_DEFINITIONS_FILE)
intervention_labels = list(intervention_definitions.keys())



# --- Helper/Utility Functions ---
def load_names_data(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Names file not found at {file_path}. Using dummy names.")
        return {
            "Male_names": {"first": ["John", "Peter", "Mike"], "last": ["Doe", "Smith", "Abila"]},
            "Female_names": {"first": ["Jane", "Mary", "Sarah"], "last": ["Doe", "Achieng", "Wanjiru"]}
        }
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {file_path}. Using dummy names.")
        return {
            "Male_names": {"first": ["John", "Peter", "Mike"], "last": ["Doe", "Smith", "Abila"]},
            "Female_names": {"first": ["Jane", "Mary", "Sarah"], "last": ["Doe", "Achieng", "Wanjiru"]}
        }

def generate_person_name(gender_str, names_data_dict):
    """Generates a first and last name based on gender."""
    if gender_str.lower() == "male":
        first_names = names_data_dict.get("Male_names", {}).get("first", ["DefaultMaleFirst"])
        last_names = names_data_dict.get("Male_names", {}).get("last", ["DefaultMaleLast"])
    elif gender_str.lower() == "female":
        first_names = names_data_dict.get("Female_names", {}).get("first", ["DefaultFemaleFirst"])
        last_names = names_data_dict.get("Female_names", {}).get("last", ["DefaultFemaleLast"])
    else: # Fallback for other gender strings or if gender is unknown
        all_first = names_data_dict.get("Male_names", {}).get("first", []) + \
                    names_data_dict.get("Female_names", {}).get("first", ["DefaultFirst"])
        all_last = names_data_dict.get("Male_names", {}).get("last", []) + \
                   names_data_dict.get("Female_names", {}).get("last", ["DefaultLast"])
        first_names = all_first if all_first else ["DefaultFirst"]
        last_names = all_last if all_last else ["DefaultLast"]

    return random.choice(first_names), random.choice(last_names)

def generate_phone_number(probability_unknown=0.15):
    """Generates a random Kenyan-style phone number or 'Unknown'."""
    if random.random() < probability_unknown:
        return "Unknown"
    prefix = random.choice(["07", "01"]) # Common prefixes
    number = "".join([str(random.randint(0, 9)) for _ in range(8)]) # 8 more digits
    return f"{prefix}{number}"

import random

def generate_prompt(case_details: dict, quality_level: str = "good"):
    # Define scenario contexts
    scenarios = {
        "child_abuse_report": {
            "context": "A caller is reporting suspected child abuse or neglect",
            "caller_emotion": "distressed, concerned, urgent",
            "agent_role": "child protection helpline counselor"
        },
        "mental_health_crisis": {
            "context": "A caller is experiencing mental health crisis or suicidal thoughts",
            "caller_emotion": "desperate, overwhelmed, sad",
            "agent_role": "crisis intervention counselor"
        },
        "domestic_violence": {
            "context": "A caller is reporting domestic violence incident",
            "caller_emotion": "fearful, anxious, seeking help",
            "agent_role": "domestic violence support counselor"
        },
        "general_inquiry": {
            "context": "A caller is asking for general information about services",
            "caller_emotion": "curious, neutral, seeking information",
            "agent_role": "information helpline agent"
        },
        "follow_up_call": {
            "context": "A follow-up call about a previous case or service",
            "caller_emotion": "mixed emotions, seeking updates",
            "agent_role": "case follow-up specialist"
        }
    }

    # Randomly choose one scenario
    scenario_type, scenario = random.choice(list(scenarios.items()))

    # Define quality level guidelines
    quality_instructions = {
        "excellent": {
            "opening": "Use proper greeting with helpline name, agent name and offer of assistance",
            "listening": "Show active listening, empathy, use polite language, no interruptions",
            "proactiveness": "Offer additional help, confirm satisfaction, mention follow-up",
            "resolution": "Provide accurate info, explain clearly, follow proper steps",
            "hold": "Explain before hold, provide updates, thank for waiting",
            "closing": "Use proper closing phrase and offer future assistance"
        },
        "good": {
            "opening": "Basic greeting with call agent name, may miss some elements",
            "listening": "Some empathy shown, mostly polite, minimal interruptions",
            "proactiveness": "Some additional help offered, basic confirmation",
            "resolution": "Generally accurate info, adequate explanations",
            "hold": "Basic hold procedures, may miss some elements",
            "closing": "Basic closing, may not offer future help"
        },
        "poor": {
            "opening": "Minimal or no proper greeting",
            "listening": "Little empathy, interruptions, impatient tone",
            "proactiveness": "Minimal extra help, no follow-up mentioned",
            "resolution": "Vague information, unclear explanations",
            "hold": "Poor hold etiquette, no explanations or thanks",
            "closing": "Abrupt ending, no proper closing"
        }
    }

    quality_level = quality_level.lower()
    quality_guide = quality_instructions.get(quality_level, quality_instructions["good"])

    # Extract and fill case details
    details = {
        "category": case_details.get("category", "Unknown Category"),
        "category_definition": case_details.get("category_definition", "Unknown Definition"),
        "victim_first_name": case_details.get("victim_first_name", "Jane"),
        "victim_last_name": case_details.get("victim_last_name", "Doe"),
        "victim_phone": case_details.get("victim_phone", "Unknown"),
        "victim_age": case_details.get("victim_age", "unknown age"),
        "victim_gender": case_details.get("victim_gender", "unknown gender"),
        "reporter_first_name": case_details.get("reporter_first_name", "John"),
        "reporter_last_name": case_details.get("reporter_last_name", "Smith"),
        "reporter_phone": case_details.get("reporter_phone", "Unknown"),
        "reporter_age": case_details.get("reporter_age", "unknown age bracket"),
        "reporter_gender": case_details.get("reporter_gender", "unknown gender"),
        "reporter_relationship": case_details.get("reporter_relationship", "unknown relationship"),
        "county": case_details.get("county", "Unknown County"),
        "subcounty": case_details.get("subcounty", "Unknown Subcounty"),
        "ward": case_details.get("ward", "Unknown Ward"),
        "landmark": case_details.get("landmark", "Unknown Landmark"),
        "perpetrator_first_name": case_details.get("perpetrator_first_name", "Alex"),
        "perpetrator_last_name": case_details.get("perpetrator_last_name", "Anonymous"),
        "perpetrator_phone": case_details.get("perpetrator_phone", "Unknown"),
        "perpetrator_age": case_details.get("perpetrator_age", "unknown age bracket"),
        "perpetrator_gender": case_details.get("perpetrator_gender", "unknown gender"),
        "perpetrator_relationship": case_details.get("perpetrator_relationship", "unknown relationship"),
    }
    interventions_list = ", ".join([f'"{label}"' for label in intervention_labels])


    return f"""
You are an experienced Kenyan child protection counselor, generating a realistic call transcript for a {scenario['agent_role']} handling a call from an East African Child helpline platform.
Your task is to generate a concise call transcript, related and affecting children and also Gender based violence  then classify the conversation transcripts into categories and specify an intervention.

Case Details:
- Category of Harm: {details["category"]}
- Victim: {details["victim_first_name"]} {details["victim_last_name"]} ({details["victim_age"]} years, {details["victim_gender"]})
- Location: {details["landmark"]}, {details["ward"]} Ward, {details["subcounty"]} Subcounty, {details["county"]} County.
- Reporter: {details["reporter_first_name"]} {details["reporter_last_name"]} ({details["reporter_relationship"]} to victim)
- Perpetrator (if applicable): {details["perpetrator_first_name"]} {details["perpetrator_last_name"]} ({details["perpetrator_relationship"]} to victim)

SCENARIO: {scenario['context']}
CALLER EMOTION: {scenario['caller_emotion']}
CALL QUALITY LEVEL: {quality_level.capitalize()}

INTERVENTION LABELS (choose one): {interventions_list}

QUALITY GUIDELINES:
- Opening: {quality_guide['opening']}
- Listening: {quality_guide['listening']}
- Proactiveness: {quality_guide['proactiveness']}
- Resolution: {quality_guide['resolution']}
- Hold: {quality_guide['hold']}
- Closing: {quality_guide['closing']}

CRITICAL FORMATTING REQUIREMENTS:
1. Generate a CONTINUOUS conversation transcript WITHOUT any speaker labels
2. DO NOT diarize or separate the individual speakers or  use "Agent:", "I", "Caller:", "Operator:" or any speaker identification
3. DO NOT include tone descriptions in parentheses like "(distressed tone)" or "(urgent tone)"
4. The conversation should read as a natural flow of dialogue
5. Use line breaks to separate different speakers' turns
6. Show emotions and tone through the actual words and speech patterns, not through labels

CONTENT REQUIREMENTS:
1. Make the conversation about 400 words and not have a token limit of 512 tokens
2. Include realistic dialogue that demonstrates the specified quality level
3. Include natural pauses, emotions, and realistic speech patterns in the actual speech
4. If quality is "poor", show unprofessional behavior through actual dialogue
5. If quality is "excellent", demonstrate best practices through actual dialogue
6. Include scenarios that would naturally trigger each QA metric to separate the two users 
7. DO NOT use "Agent:", "I", "Caller:", "Operator:" or any names or any speaker identification
8. DO NOT use tone descriptions in parentheses like "(distressed tone)" or "(urgent tone)"
9. The conversation should flow naturally, and it should not be diarized, it should be a continuous conversation with no tags for caller or agent.
10. Output just the conversation transcript without any additional formatting or tags

<thinking>
1. Generate the Transcript:
- Generate a call transcript basing the core issue on the Category: "{details["category"]}".
- Identify key individuals and their roles.
- For more rich data include the case details meticulously i.e {details["victim_first_name"]}.

2. Construct Narrative:
- Briefly write down the transcript, mimicking a call conversation between the helpline agent and the caller. The narrative should be related to  "{details["category"]}". Include who, what, when, where, and how mimicking a live call between an agent and a caller.

3. Format Output (Strictly follow this format):
- The "Text" field should contain a transcript(400 words or 512 tokens).
- The "Casetype" field should be exactly: "{details["category"]}"
- The "Intervention" field should contain one of the intervention labels from the list provided above.
- The "Priority" field should be a digit value of 1, 2, or 3 based on the call narrative.

4. DO NOT AT ANY POINT DIARIZE OR SEPARATE INTO INDIVIDUAL SPEAKERS THE TRANSCRIPT INTO INDIVIDUAL SPEAKERS

EXAMPLE OF CORRECT FORMAT:
Hello, thank you for calling Crisis Support Services. My name is Sarah. How can I help you today?
Hi, I... I really need someone to talk to. I'm having a really hard time right now.
I'm here to listen. Can you tell me what's going on?
I just feel so overwhelmed. Everything seems to be falling apart and I don't know what to do.

EXAMPLE OF INCORRECT FORMAT (DO NOT DO THIS):
Agent (professional tone): Hello, thank you for calling Crisis Support Services...
Caller (distressed): Hi, I really need someone to talk to...

OUTPUT FORMAT:
Generate ONLY the conversation transcript in the correct continuous format described above, then analyse the call transcript to prescribe and recommend the best course of action or intervention needed for the case.
No additional formatting, headers, or explanations. Keep in mind of the example of correct format provided.

Priority:
Based on the call narrative, analyse the conversation and prioritize it into one of the 3 levels; 1, 2,3. 

RETURN ONLY THE priority DIGIT VALUE, DO NOT ADD ANY TEXT OR LABELS.
</thinking>


CRITICAL OUTPUT FORMAT REQUIREMENTS:
You MUST output EXACTLY and ONLY in this format:
<text>
[Generated transcript text here]
</text>
<label>
{details["category"]}
</label>
<intervention>
[One intervention label from the list provided above]
</intervention>
<priority>
[One of: 1, 2, 3 based on the call narrative]
</priority>

VALIDATION:
Ensure the output is exactly as specified.
Ensure the conversation narrative should contain a transcript of about 400 words or 512 tokens.

Follow the exact output format meticulously, including ALL the tags and structure.

<text>
[Generated transcript text here]
</text>
<label>
{details["category"]}
</label>
<intervention>
[One intervention label from the list above]
</intervention>
<priority>
[One of: 1, 2, or 3 based on the call narrative]
</priority>
"""


# use ollama 

def call_mistral_llm(prompt):
    """Call Mistral LLM via Ollama REST API to generate transcript"""
    payload = {
        "model": "mistral",  # or your specific model name in Ollama
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",  # or your Ollama server address
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        llm_response = response.json().get("response", "")
        transcript = llm_response.strip()
        logging.info("✅ LLM transcript generated via Ollama.")
        return transcript
    except Exception as e:
        logging.error(f"❌ Failed to generate transcript via Ollama: {e}")
        return None

def generate_synthetic_data(num_years=1, num_cases=100):
    names_data_loaded = load_names_data(NAMES_FILE) 

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * num_years)
    end_timestamp, start_timestamp = int(end_date.timestamp()), int(start_date.timestamp())

    data = []
    category_counts = {category: 0 for category in CATEGORIES_LABELS}

    try:
        with open(LOCATIONS_FILE, "r", encoding="utf-8") as fp:
            loc_data_raw = json.load(fp)
            # Attempt to infer structure or use a known structure
            if isinstance(loc_data_raw, list) and len(loc_data_raw) > 0 and isinstance(loc_data_raw[0], dict):
                locdf = pd.DataFrame(loc_data_raw)
            elif isinstance(loc_data_raw, dict):
                locdf = pd.DataFrame(loc_data_raw) 
            else:
                raise ValueError("Unexpected JSON structure in locations file.")
        logging.info(f"Locations loaded, columns: {locdf.columns.tolist()}")

    except FileNotFoundError:
        logging.warning(f"Location file not found: {LOCATIONS_FILE}. Using dummy locations.")
        locdf = pd.DataFrame({
            "county": ["County1"], "district": ["District1"],
            "ward": ["Ward1"], "station": ["Station1"]
        })

    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"Error processing location file {LOCATIONS_FILE}: {e}. Using dummy locations.")
        locdf = pd.DataFrame({
            "county_name": ["Nairobi"], "sub_county_name": ["Starehe"],
            "ward_name": ["Central"], "polling_station_name": ["City Hall"]
        })
    rename_map = {
        'county': 'county_name', 'County': 'county_name',
        'district': 'sub_county_name', 'subcounty': 'sub_county_name', 'SubCounty': 'sub_county_name',
        'ward': 'ward_name', 'Ward': 'ward_name',
        'station': 'polling_station_name', 'landmark': 'polling_station_name', 'Polling Station': 'polling_station_name'
    }
    # Create a new map with only existing columns to avoid errors
    actual_rename_map = {k: v for k, v in rename_map.items() if k in locdf.columns}
    locdf = locdf.rename(columns=actual_rename_map)

    # Define the expected columns after renaming
    expected_loc_cols = {'county_name', 'sub_county_name', 'ward_name', 'polling_station_name'}
    missing_cols = expected_loc_cols - set(locdf.columns)
    if missing_cols:
        logging.warning(f"Location data is missing expected columns: {missing_cols}. Defaults will be used for these.")
        for col in missing_cols:
            locdf[col] = f"Unknown {col.replace('_name', '')}"


    DROP = ["loop", "idcounty", "iddistrict", "idward", "streams"]
    locdf = locdf.drop(columns=[col for col in DROP if col in locdf.columns], errors='ignore')

    gender_options = ["female"] * 7 + ["male"] * 3 # 70% female victims
    perp_gender_options = ["male"] * 8 + ["female"] * 2 # 80% male perpetrators


    work_hours_start, work_hours_end = 6 * 3600, 22 * 3600 # 6 AM to 10 PM
    day_of_week_counts = {i: 0 for i in range(7)}
    time_periods = [(6*3600,9*3600), (9*3600,12*3600), (12*3600,15*3600), (15*3600,18*3600), (18*3600,20*3600), (20*3600,22*3600)]
    time_weights = [10,15,20,30,40,25]  # Peak mid-day to early evening
    reporter_age_brackets = ["Child (under 18)","Young Adult (18-24)","Adult (25-40)","Middle-aged (41-60)","Senior (61+)","Unknown"]
    reporter_weights = [5,15,40,30,5,5] # More adult reporters
    perpetrator_age_brackets = ["Child (under 12)","Adolescent (12-17)","Young Adult (18-24)","Adult (25-40)","Middle-aged (41-60)","Senior (61+)","Unknown"]
    perpetrator_weights = [3,7,25,40,20,3,2] # Mostly adult perpetrators
    reporter_relationship_options = ["Parent","Guardian","Grandparent","Teacher/School Staff","Neighbor","Relative","Sibling","Social Worker","Medical Professional","Friend of Family","Community Member","Self-reporting","Anonymous"]
    reporter_relationship_weights = [30,15,10,12,7,8,5,10,8,6,5,2,8]
    perpetrator_relationship_options = ["Parent","Step-parent","Guardian","Relative","Sibling","Family Friend","Teacher","Caregiver","Neighbor","Stranger","Coach/Instructor","Religious Leader","Peer/Classmate","Employer","Unknown"]
    perpetrator_relationship_weights = [25,15,10,12,8,10,5,7,6,5,3,2,8,4,5]

    for x in range(num_cases):
        logging.info(f"--- Generating case {x+1}/{num_cases} ---")
        dat = {}

        location_info = {}
        try:
            if not locdf.empty:
                 # Ensure sampling from rows with non-missing values for key fields if possible
                valid_rows = locdf.dropna(subset=['county_name', 'sub_county_name', 'ward_name', 'polling_station_name'])
                if not valid_rows.empty:
                    location_sample = valid_rows.sample().iloc[0]
                else:
                    location_sample = locdf.sample().iloc[0] # Fallback to any row
                 
                location_info = {
                    "county": location_sample.get("county_name", "Unknown County"),
                    "subcounty": location_sample.get("sub_county_name", "Unknown Subcounty"),
                    "ward": location_sample.get("ward_name", "Unknown Ward"),
                    "landmark": location_sample.get("polling_station_name", "Unknown Landmark")
                }
            else: raise ValueError("Location dataframe is empty.")
        except Exception as e_loc: # Catch more general errors during sampling
            logging.warning(f"Error getting location info: {e_loc}. Using default location.")
            location_info = {"county": "Default County", "subcounty": "Default Subcounty", "ward": "Default Ward", "landmark": "Default Landmark"}



        random_event_timestamp = random.randint(start_timestamp, end_timestamp)
        dt_event = datetime.fromtimestamp(random_event_timestamp)
        
        day_of_week = dt_event.weekday()
        if day_of_week < 5 and random.random() < 0.4:
            for _ in range(3):
                new_ts = random.randint(start_timestamp, end_timestamp)
                new_dt = datetime.fromtimestamp(new_ts)
                if new_dt.weekday() >= 5:
                    dt_event, random_event_timestamp = new_dt, new_ts
                    break
        day_of_week_counts[dt_event.weekday()] += 1
        
        dt_call_day_midnight = dt_event.replace(hour=0, minute=0, second=0, microsecond=0)
        midnight_timestamp_call_day = int(dt_call_day_midnight.timestamp())
        selected_time_period = random.choices(time_periods, weights=time_weights, k=1)[0]
        seconds_into_call_day = random.randint(selected_time_period[0], selected_time_period[1])
        call_start_timestamp = max(start_timestamp, min(midnight_timestamp_call_day + seconds_into_call_day, end_timestamp))

        current_time_for_id = str(call_start_timestamp)
        dat["uniqueid"] = current_time_for_id + "." + str(random.randint(1000, 9999))
        dat["transcription"], dat["translation"] = {}, {}

        dt_call_start_obj = datetime.fromtimestamp(call_start_timestamp)
        dat["startdate"] = dt_call_start_obj.strftime("%d %b %Y")
        dat["starttime"] = dt_call_start_obj.strftime("%H:%M:%S")

        seconds_since_midnight_for_call = dt_call_start_obj.hour*3600 + dt_call_start_obj.minute*60 + dt_call_start_obj.second
        talk_seconds = random.randint(10*60, 20*60)
        if seconds_since_midnight_for_call + talk_seconds > work_hours_end:
            talk_seconds = max(0, work_hours_end - seconds_since_midnight_for_call)
        unixt_call_end = call_start_timestamp + talk_seconds
        dat["stoptime"] = datetime.fromtimestamp(unixt_call_end).strftime("%H:%M:%S")
        minutes, seconds = divmod(talk_seconds, 60)
        dat["talktime"] = f"{minutes}:{seconds:02d}"

        # Category 
        sampled_category = random.choice(CATEGORIES_LABELS)
        dat["category"] = sampled_category
        category_counts[sampled_category] += 1
        dat["category_definition"] = category_definitions_data.get(sampled_category, "Unknown Definition")

        case_details_for_prompt = {"category": sampled_category}



        # Victim details
        dat["victim"] = {}
        dat["victim"]["gender"] = case_details_for_prompt["victim_gender"] = random.choice(gender_options)
        dat["victim"]["first_name"], dat["victim"]["last_name"] = generate_person_name(dat["victim"]["gender"], names_data_loaded)
        case_details_for_prompt["victim_first_name"] = dat["victim"]["first_name"]
        case_details_for_prompt["victim_last_name"] = dat["victim"]["last_name"]
        dat["victim"]["phone_number"] = case_details_for_prompt["victim_phone"] = generate_phone_number()
        victim_age_val = random.randint(3, 16)
        dat["victim"]["age"] = case_details_for_prompt["victim_age"] = str(victim_age_val)
        dat["victim"]["birthday"] = str(dt_event.year - victim_age_val)

        # Reporter details
        dat["reporter"] = {}
        dat["reporter"]["gender"] = case_details_for_prompt["reporter_gender"] = random.choice(gender_options)
        dat["reporter"]["first_name"], dat["reporter"]["last_name"] = generate_person_name(dat["reporter"]["gender"], names_data_loaded)
        case_details_for_prompt["reporter_first_name"] = dat["reporter"]["first_name"]
        case_details_for_prompt["reporter_last_name"] = dat["reporter"]["last_name"]
        dat["reporter"]["phone_number"] = case_details_for_prompt["reporter_phone"] = generate_phone_number()
        dat["reporter"]["age"] = case_details_for_prompt["reporter_age"] = random.choices(reporter_age_brackets, weights=reporter_weights, k=1)[0]
        dat["reporter"]["relationship"] = case_details_for_prompt["reporter_relationship"] = random.choices(reporter_relationship_options, weights=reporter_relationship_weights, k=1)[0]
        
        # Location details
        dat["county"] = case_details_for_prompt["county"] = location_info.get("county", "Unknown County")
        dat["subcounty"] = case_details_for_prompt["subcounty"] = location_info.get("district", "Unknown Subcounty")
        dat["ward"] = case_details_for_prompt["ward"] = location_info.get("ward", "Unknown Ward")
        dat["landmark"] = case_details_for_prompt["landmark"] = location_info.get("station", "Unknown Landmark")

        # Perpetrator details
        dat["perpetrator"] = {}
        dat["perpetrator"]["gender"] = case_details_for_prompt["perpetrator_gender"] = random.choice(perp_gender_options)
        dat["perpetrator"]["first_name"], dat["perpetrator"]["last_name"] = generate_person_name(dat["perpetrator"]["gender"], names_data_loaded)
        case_details_for_prompt["perpetrator_first_name"] = dat["perpetrator"]["first_name"]
        case_details_for_prompt["perpetrator_last_name"] = dat["perpetrator"]["last_name"]
        dat["perpetrator"]["phone_number"] = case_details_for_prompt["perpetrator_phone"] = generate_phone_number(probability_unknown=0.3) # Higher chance of unknown for perp
        dat["perpetrator"]["age"] = case_details_for_prompt["perpetrator_age"] = random.choices(perpetrator_age_brackets, weights=perpetrator_weights, k=1)[0]
        dat["perpetrator"]["relationship"] = case_details_for_prompt["perpetrator_relationship"] = random.choices(perpetrator_relationship_options, weights=perpetrator_relationship_weights, k=1)[0]
        
        dat["counselor"] = "Counselor " + random.choice(string.ascii_uppercase)

        # LLM Narrative Generation
        prompt_text = generate_prompt(case_details_for_prompt)
        logging.info(f"Prompt for LLM: Cat: {sampled_category},  Victim: {dat['victim']['first_name']}")
        # print(f"DEBUG PROMPT:\n{prompt_text}\n") 
        
        llm_full_output_str = call_mistral_llm(prompt_text)
        print(f"DEBUG LLM OUTPUT:\n{llm_full_output_str}\n") # Debugging output
        if llm_full_output_str:
            try:
                # Extract text content
                text_match = re.search(r"<text>(.*?)</text>", llm_full_output_str, re.DOTALL | re.IGNORECASE)
                if text_match:
                    text_content = text_match.group(1).strip()
                    text_content = re.sub(r'\n\s+', '\n', text_content)
                else:
                    text_content = "Narrative not extracted."
                    logging.warning("⚠️ <text> tags not found in LLM output")

                # Extract label content
                label_match = re.search(r"<label>(.*?)</label>", llm_full_output_str, re.DOTALL | re.IGNORECASE)
                if label_match:
                    label_content = label_match.group(1).strip()
                else:
                    label_content = sampled_category
                    logging.warning(f"⚠️ <label> tags not found, using original category: {sampled_category}")

                # Extract intervention content
                intervention_match = re.search(r"<intervention>(.*?)</intervention>", llm_full_output_str, re.DOTALL | re.IGNORECASE)
                if intervention_match:
                    intervention_content = intervention_match.group(1).strip()
                else:
                    intervention_content = ""
                    logging.warning("⚠️ <intervention> tags not found in LLM output")
                
                # Extract priority content
                priority_match = re.search(r"<priority>\s*([123])\s*</priority>", llm_full_output_str, re.DOTALL | re.IGNORECASE)
                if not priority_match:
                    # Try to match <priority>\s*([123])\s*($|\n)
                    priority_match = re.search(r"<priority>\s*([123])\s*(?:</priority>|$|\n)", llm_full_output_str, re.IGNORECASE)
                if priority_match:
                    priority_content = priority_match.group(1).strip()
                else:
                    priority_content = "N/A"
                    logging.warning("⚠️ <priority> tags not found or malformed in LLM output")

                                           

                dat["narrative"] = text_content
                dat["category"] = label_content
                dat["intervention"] = intervention_content
                dat["priority"] = priority_content
                logging.info(f"Narrative: {text_content[:70]}...")

            except Exception as e_parse:
                logging.error(f"❌ Error parsing LLM output: {e_parse}")
                dat["narrative"] = "Error parsing LLM output."
                dat["category"] = sampled_category
                dat["intervention"] = "N/A"
                dat["priority"] = "N/A"
        #     except Exception as e_parse:
        #         logging.error(f"Error parsing LLM output: {e_parse}. Raw: {llm_full_output_str[:150]}...")
        #         dat["narrative"] = "Error parsing LLM output."
        #         dat["category"] = sampled_category
        # else:
        #     dat["narrative"] = "Narrative generation failed (LLM call error)."
        #     dat["category"] = sampled_category

      
        data.append(dat)
        
    # Print Metrics

    print("\n--- Metrics ---")
    print("Day of week distribution (0=Mon, 6=Sun):")
    days_map = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    total_cases_for_metrics = num_cases if num_cases > 0 else 1 # Avoid division by zero
    for day_num, count in day_of_week_counts.items():
        print(f"{days_map[day_num]}: {count} ({count/total_cases_for_metrics*100:.1f}%)")
        
    reporter_rel_counts = {}
    perpetrator_rel_counts = {}
    for case_item in data:
        if "reporter" in case_item and "relationship" in case_item["reporter"]:
             rep_rel = case_item["reporter"]["relationship"]
             reporter_rel_counts[rep_rel] = reporter_rel_counts.get(rep_rel, 0) + 1
        if "perpetrator" in case_item and "relationship" in case_item["perpetrator"]:
             perp_rel = case_item["perpetrator"]["relationship"]
             perpetrator_rel_counts[perp_rel] = perpetrator_rel_counts.get(perp_rel, 0) + 1
    
    if reporter_rel_counts:
        print("\nReporter relationship distribution:")
        for rel, count in sorted(reporter_rel_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{rel}: {count} ({count/total_cases_for_metrics*100:.1f}%)")
    
    if perpetrator_rel_counts:
        print("\nPerpetrator relationship distribution:")
        for rel, count in sorted(perpetrator_rel_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{rel}: {count} ({count/total_cases_for_metrics*100:.1f}%)")

    print("\nCategory distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{cat}: {count} ({count/total_cases_for_metrics*100:.1f}%)")
    

    
    return data


if __name__ == "__main__":

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logging.info(f"Created directory: {DATA_DIR}")


    # Ensure definition files exist or create dummies
    for file_path, def_type, dummy_data in [
        (CATEGORIES_DEFINITIONS_FILE, "category", {"Generic Harm": "A generic harm to a child."})
    ]:
        if not os.path.exists(file_path):
            logging.warning(f"CRITICAL: {def_type.capitalize()} definitions file not found at {file_path}. Creating a dummy file.")
            with open(file_path, "w", encoding="utf-8") as f_dummy:
                json.dump(dummy_data, f_dummy, indent=4)
            # Re-initialize globals if they were set before this check and file was created
            if def_type == "category":
                category_definitions_data = load_definitions_from_json(CATEGORIES_DEFINITIONS_FILE, "category")
                CATEGORIES_LABELS = list(category_definitions_data.keys())
            

    num_years_to_generate = 5
    num_cases_to_generate = 3
    
    
    logging.info(f"Starting data generation for {num_cases_to_generate} cases / {num_years_to_generate} year(s).")
    cases_data = generate_synthetic_data(num_years=num_years_to_generate, num_cases=num_cases_to_generate)
    df = pd.DataFrame(cases_data)
    df.to_csv("cases_generated_data_v0005.csv", index=False, encoding="utf-8")
    logging.info(f"✅ Exported {len(cases_data)} cases to CSV: cases_generated_data_v0005.csv")

    if cases_data:
        output_dir = os.path.join(BASE_DIR, "casedir")
        output_file = os.path.join(output_dir, "cases_generated_data_v0005.json")
        logging.info(f"Generated {len(cases_data)} cases. Saved to {output_file}")

        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(output_file, "w") as fp:
                json.dump(cases_data, fp, indent=4)
            logging.info(f"Generated {len(cases_data)} cases. Saved to {output_file}")
        except Exception as e:
            logging.error(f"Error saving data: {e}")
            fallback_file = "cases_generated_data_v10_fallback.json"
            with open(fallback_file, "w") as fp:
                json.dump(cases_data, fp, indent=4)
            logging.info(f"Saved to fallback: {fallback_file}")
    else:
        logging.warning("No data was generated.")
