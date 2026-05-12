import re
import spacy
import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util

# Load once (GLOBAL)
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

abbr_df = pd.read_csv("data/abbreviations_data.csv")
SKILLS_DB = pd.read_csv("data/skills_data.csv")["skill"].tolist()

with open("data/skill_categories.json") as f:
    SKILL_CATEGORIES = json.load(f)

# Lookup
CATEGORY_LOOKUP = {
    skill: category
    for category, skills in SKILL_CATEGORIES.items()
    for skill in skills
}

ABBREVIATIONS = dict(zip(abbr_df["abbv"], abbr_df["term"]))

ABBR_PATTERN = re.compile(
    r'\b(' + '|'.join(map(re.escape, ABBREVIATIONS.keys())) + r')\b'
)

def build_skill_pattern(skills):
    skills = sorted(skills, key=len, reverse=True)
    return re.compile(r'(?<!\w)(' + '|'.join(map(re.escape, skills)) + r')(?!\w)')

SKILL_PATTERN = build_skill_pattern(SKILLS_DB)

skill_embeddings = model.encode(SKILLS_DB, convert_to_tensor=True)

# Normalize text, expand abbreviations, and remove noise
def clean_text(text):
    text = text.lower()

    text = ABBR_PATTERN.sub(lambda x: ABBREVIATIONS[x.group()], text)

    text = re.sub(r'[^\w\s+#.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Split raw text into sentences and clean each sentence
def get_sentences(text):
    doc = nlp(text)  

    sentences = [  
        clean_text(sent.text)    
        for sent in doc.sents  
        if sent.text.strip()  
    ]  
  
    return sentences


# Compute sentence-level similarity between resume and JD
def sentence_level_similarity(resume, jd):
    resume_sentences = get_sentences(resume)
    jd_sentences = get_sentences(jd)

    if not resume_sentences or not jd_sentences:  
        return 0  

    resume_embeddings = model.encode(resume_sentences, convert_to_tensor=True)  
    jd_embeddings = model.encode(jd_sentences, convert_to_tensor=True)  

    sim_matrix = util.cos_sim(resume_embeddings, jd_embeddings)  

    max_scores = sim_matrix.max(dim=0).values  

    filtered_scores = [s.item() for s in max_scores if s > 0.4]
    return sum(filtered_scores) / len(filtered_scores) if filtered_scores else 0


# Extract exact skill matches using regex
def extract_skills_regex(text):
    text = clean_text(text)
    matches = re.findall(SKILL_PATTERN, text)
    return set(matches)


# Extract meaningful noun phrases for semantic matching
def extract_phrases_spacy(text):
    doc = nlp(text)
    
    phrases = set()
    
    # capture noun chunks as candidate skills
    for chunk in doc.noun_chunks:

        # skip chunks with only stopwords
        if all(token.is_stop for token in chunk):
            continue
            
        lemmatized = " ".join([token.lemma_ for token in chunk])
        phrases.add(lemmatized.lower())
    
    return list(phrases)


# Match phrases to skills using embedding similarity
def semantic_skill_extractor(text, base_threshold):
    phrases = extract_phrases_spacy(text)

    if not phrases:
        return set()

    phrase_embeddings = model.encode(phrases, convert_to_tensor=True)
    sim_matrix = util.cos_sim(phrase_embeddings, skill_embeddings)

    found = set()

    for i in range(sim_matrix.shape[0]):
        scores = sim_matrix[i]

        # Get top-k matches
        max_score, max_idx = scores.max(dim=0)

        # Check threshold
        if max_score.item() >= base_threshold:
            found.add(SKILLS_DB[max_idx])

    return found


# Combine regex and semantic skill extraction
def hybrid_skill_extraction(text):

    regex_skills = extract_skills_regex(text)  
    semantic_skills = semantic_skill_extractor(text, 0.8)  
  
    # Merge all  
    final_skills = regex_skills | semantic_skills 
  
    return final_skills


# Compute overlap ratio between resume and JD skills
def skill_match_score(resume_skills, jd_skills):
    if not jd_skills:
        return 0
    return len(resume_skills & jd_skills) / len(jd_skills)


# Map extracted skills to their categories
def map_to_category(skills):
    mapped = set()

    for skill in skills:
        mapped.add(skill)
        if skill in CATEGORY_LOOKUP:
            mapped.add(CATEGORY_LOOKUP[skill]) 

    return mapped


# End-to-end resume vs JD evaluation
def final_similarity(resume, jd):

    resume_skills = hybrid_skill_extraction(resume)  
    jd_skills = hybrid_skill_extraction(jd)  

    resume_skills = map_to_category(resume_skills)
    jd_skills = map_to_category(jd_skills)
    
    skill_score = skill_match_score(resume_skills, jd_skills)  
    semantic_score = sentence_level_similarity(resume, jd)  

    final_score = (0.3 * semantic_score) + (0.7 * skill_score)  

    return {
        "matched_skills": list(resume_skills & jd_skills),
        "missing_skills": list(jd_skills - resume_skills),
        "skill_score": round(skill_score, 2)*100,
        "final_score": round(final_score, 2)*100
    }