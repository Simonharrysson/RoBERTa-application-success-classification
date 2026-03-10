import pandas as pd
from transformers import pipeline, AutoTokenizer
import re
from datasets import load_dataset


gender_words = {
    'female': ['woman', 'girl', 'lady', 'mother', 'wife', 'daughter', 'sister', 'mom'],
    'male': ['man', 'boy', 'boys', 'gentleman', 'father', 'husband', 'son', 'brother', 'dad']
}

tokenizer = AutoTokenizer.from_pretrained("ab-ai/pii_model_based_on_distilbert")

ner_pipeline = pipeline("ner",
                        model="ab-ai/pii_model_based_on_distilbert",
                        aggregation_strategy="simple",
                        device=-1)


def main():
    dataset = load_dataset("facehuggerapoorv/resume-jd-match")
    data = dataset["test"].to_pandas()
    common_names = pd.read_csv("utils/common_names.csv")
    gender_neutral_job_titles = pd.read_csv("raw_data/gender_neutral_job_titles.csv")
    data["Resume"] = data["text"].apply(extract_resume)
    data[["Resume"]].to_csv("raw_data/resumes.csv", index=False)

    resumes = pd.read_csv("raw_data/first_resume.csv")
    extracted_info_dict = {}

    # Anonymization
    for idx, row in resumes.iterrows():
        text = row['Resume']
        text = anonymize_names(text, idx, extracted_info_dict)
        text = anonymize_emails(text, idx, extracted_info_dict)
        text = anonymize_usernames(text, idx, extracted_info_dict)
        identify_gender(idx, extracted_info_dict, common_names)
        text = anonymize_gender_words(text, idx, extracted_info_dict)
        text = anonymize_gender_job_titles(text, idx, gender_neutral_job_titles, extracted_info_dict)
        resumes.at[idx, 'anonymized_text'] = text

    resumes.to_csv("anonymized_files/anonymized_file.csv", index=False)
    pd.DataFrame.from_dict(extracted_info_dict, orient='index').to_csv("anonymized_files/dictionary.csv")


def extract_resume(text):
    match = re.search(r'the resume: <<(.+?)>>\. The result is', text, re.DOTALL)
    return match.group(1).strip() if match else None


def predict_entities_chunked(text, chunk_size=400, overlap=50):
    """ Run NER pipeline on long text by chunking based on actual tokens."""
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoding['input_ids']
    offsets = encoding['offset_mapping']

    all_entities = []
    start = 0

    while start < len(input_ids):
        end = min(start + chunk_size, len(input_ids))
        chunk_ids = input_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids)
        char_offset = offsets[start][0]

        results = ner_pipeline(chunk_text)
        for entity in results:
            entity['start'] += char_offset
            entity['end'] += char_offset
            all_entities.append(entity)

        start += chunk_size - overlap

    return all_entities


def indentified_name(index, extracted_info_dict):
    if not extracted_info_dict[index].get('name'):
        return False
    return True


def anonymize_names(text, index, extracted_info_dict):
    if not isinstance(text, str):
        return text

    if index not in extracted_info_dict:
        extracted_info_dict[index] = {'name': []}

    entities = predict_entities_chunked(text)
    entities = sorted(entities, key=lambda x: x['start'])

    name_parts = {}
    for entity in entities:
        if entity['entity_group'] in ('FIRSTNAME', 'LASTNAME'):
            if entity['entity_group'] not in name_parts:  # only first occurrence of each type
                name_parts[entity['entity_group']] = entity['word']

    if name_parts and not extracted_info_dict[index]['name']:
        order = ['FIRSTNAME', 'MIDDLENAME', 'LASTNAME']
        full_name = ' '.join(name_parts[k] for k in order if k in name_parts)
        extracted_info_dict[index]['name'].append(full_name)
        extracted_info_dict[index]['name'] = [
            re.sub(r'[\s:,.\-_|]+$', '', name) for name in extracted_info_dict[index]['name']]

    for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
        if entity['entity_group'] in ('FIRSTNAME', 'LASTNAME'):
            if text[entity['start']:entity['end']].startswith('['):
                continue
            text = text[:entity['start']] + '[PERSON]' + text[entity['end']:]

    return text


def anonymize_emails(text, index, extracted_info_dict):
    """ Anonymize email addresses in the text. """
    if not isinstance(text, str):
        return text

    email_pattern = r'[\w\.-]+@[\w\.-]+'
    emails = re.findall(email_pattern, text)

    if not indentified_name(index, extracted_info_dict):
        return text

    if index not in extracted_info_dict:
        extracted_info_dict[index] = {'emails': []}

    if emails:
        extracted_info_dict[index]['emails'] = []
        for email in emails:
            if email not in extracted_info_dict[index]['emails']:
                extracted_info_dict[index]['emails'] = email
        text = re.sub(r'(?<!\[)' + r'[\w\.-]+@[\w\.-]+' + r'(?!\])', '[EMAIL]', text)

    return text


def anonymize_usernames(text, index, extracted_info_dict):
    """ Anonymize usernames in the text, e.g. Bob Smidth -> bobsmidth. """
    if not isinstance(text, str):
        return text

    if not indentified_name(index, extracted_info_dict):
        return text

    if index not in extracted_info_dict:
        extracted_info_dict[index] = {'username': []}

    name = extracted_info_dict[index]['name'][0].split()[0]
    no_space = re.escape(re.sub(r'[\s-]', '', name))
    dashed = re.escape(name.lower().replace(" ", "-"))
    pattern = re.compile(f"{no_space}|{dashed}", re.IGNORECASE)

    # Remove entire strings (URLs/handles) containing the username
    url_pattern = re.compile(r'(?<!\[)\S*(?:' + f"{no_space}|{dashed}" + r')\S*(?!\])', re.IGNORECASE)

    if re.search(pattern, text):
        extracted_info_dict[index]['username'] = re.sub(r'[\s-]', '', name).lower()
        text = re.sub(url_pattern, '[USERNAME]', text)

    return text


def identify_gender(index, extracted_info_dict, common_names):
    """ Identify gender from the first name. """

    if not extracted_info_dict[index]['name']:
        extracted_info_dict[index]['gender'] = 'unknown'
        return

    if not indentified_name(index, extracted_info_dict):
        return

    full_name = extracted_info_dict[index]['name'][0]
    first_name = full_name.split()[0].lower()

    matches = common_names[common_names['Name'].str.lower() == first_name]

    if matches.empty:
        extracted_info_dict[index]['gender'] = 'unknown'
        return

    male_prob = matches[matches['Gender'] == 'M']['Probability'].sum()
    female_prob = matches[matches['Gender'] == 'F']['Probability'].sum()

    if len(matches) == 1:
        extracted_info_dict[index]['gender'] = matches.iloc[0]['Gender']
    else:
        if (female_prob == 0 and male_prob == 0) or len(full_name) <= 2:
            extracted_info_dict[index]['gender'] = 'unknown'
        elif female_prob == 0:
            extracted_info_dict[index]['gender'] = 'M'
        elif male_prob == 0:
            extracted_info_dict[index]['gender'] = 'F'
        elif male_prob / female_prob >= 2:
            extracted_info_dict[index]['gender'] = 'M'
        elif female_prob / male_prob >= 2:
            extracted_info_dict[index]['gender'] = 'F'
        else:
            extracted_info_dict[index]['gender'] = 'unisex'


def anonymize_gender_words(text, index, extracted_info_dict):
    """ Anonymize gender words that appear next to punctuation or newlines. """
    if not isinstance(text, str):
        return text

    extracted_info_dict[index]['gender_words'] = []

    for gender, words in gender_words.items():
        for word in words:
            pattern = re.compile(
                r'(?<=[,.\-\n])' + re.escape(word) + r'(?=[,.\-\n]|$)', re.IGNORECASE)
            matches = re.findall(pattern, text)
            if matches:
                extracted_info_dict[index]['gender_words'].extend(matches)
                text = re.sub(pattern, '[GENDER]', text)

    return text


def anonymize_gender_job_titles(text, index, job_titles_list, extracted_info_dict):
    """ Replace gendered job titles with gender-neutral equivalents. """
    if not isinstance(text, str):
        return text

    if 'neutralized_titles' not in extracted_info_dict[index]:
        extracted_info_dict[index]['neutralized_titles'] = []

    for gendered, neutral in job_titles_list.items():
        pattern = re.compile(r'\b' + re.escape(gendered) + r'\b', re.IGNORECASE)
        if re.search(pattern, text):
            extracted_info_dict[index]['neutralized_titles'].append(gendered)
            text = re.sub(pattern, neutral, text)

    return text


main()