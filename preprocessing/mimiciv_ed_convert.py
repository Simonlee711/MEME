# %% 

import pandas as pd 
import os 
from tqdm import tqdm
import json 
"""
This program converts the MIMIC-IV-ED database to text for analysis using LLMs and transformers
"""
# %%
# NOTE: Modify this to the root directory of MIMIC-IV-ED
path = '/mnt/bigdata/compmed/physionet/mimic-iv-ed-2.2/mimic-iv-ed-2.2/'
# %%
def stringify_edstays(
        subject_id, 
        gender, race, anchor_age,
        arrival_transport,
        intime, outtime, 
        disposition,
        hadm_id=None,
        admittime=None, 
        dischtime=None, 
        discharge_location=None, 
        insurance=None, 
        language=None, 
        marital_status=None, 
        hospital_expire_flag=None,
        dod=None):
    response_dict = {"arrival": [], "eddischarge": [], "admission": [], "discharge": []}
    sex = 'female' if gender == 'F' else 'male'
    response_dict['arrival'].append(f"Patient {subject_id}, a {anchor_age} year old {race.lower()} {sex},")
    response_dict['arrival'].append(f"arrived via {arrival_transport.lower()} at {str(intime)}.")
    if marital_status is not None and str(marital_status) != 'nan':
        response_dict['arrival'].append(f"The patient's marital status is {marital_status.lower()}.")
    
    if insurance is not None and str(insurance) != 'nan':
        response_dict['arrival'].append(f"The patient's insurance is {insurance.lower()}.")
    
    if language is not None and str(language) != 'nan':
        response_dict['arrival'].append(f"The patient's language is {language.lower()}.")
    
    response_dict['eddischarge'].append(f"The ED disposition was {disposition.lower()} at {str(outtime)}.")
    if dod is not None and str(dod) != 'nan':
        response_dict['eddischarge'].append(f'The patient died on {str(dod).lower()}.')
    # FOR NOW, ignore the hospitalization information
    if hadm_id is not None and str(hadm_id) != 'nan':

        response_dict['admission'].append(f"The patient was admitted at {admittime.lower()}.")
        if discharge_location is None or str(discharge_location) == 'nan':
            discharge_location = 'UNKNOWN'
        response_dict['discharge'].append(f"The patient's discharge disposition was: {discharge_location.lower()} at {dischtime.lower()}.")

        if hospital_expire_flag == 1:
            response_dict['discharge'].append(f'The patient died in the hospital.')
        if dod is not None and str(dod) != 'nan':
            response_dict['discharge'].append(f'The patient died on {str(dod).lower()}.')
    else:
        response_dict['admission'].append("The patient was not admitted.")
        response_dict['discharge'].append("The patient was not admitted.")
    
    return {k: " ".join(v) for k, v in response_dict.items()}

def stringify_triage(temperature, heartrate, resprate, o2sat, sbp, dbp, pain, acuity, chiefcomplaint):
    procstring = lambda s: str(s).lower() if s is not None and str(s) != 'nan' else "not recorded"
    response_dict = {"triage": ["At triage:"]}
    response_dict['triage'].append(f"temperature was {procstring(temperature)},")
    response_dict['triage'].append(f"pulse was {procstring(heartrate)},")
    response_dict['triage'].append(f"respirations was {procstring(resprate)},")
    response_dict['triage'].append(f"o2 saturation was {procstring(o2sat)},")
    response_dict['triage'].append(f"systolic blood pressure was {procstring(sbp)},")
    response_dict['triage'].append(f"diastolic blood pressure was {procstring(dbp)},")
    response_dict['triage'].append(f"pain was {procstring(pain)},")
    response_dict['triage'].append(f"chief complaint was {procstring(chiefcomplaint)}.")
    response_dict['triage'].append(f"Acuity score was {procstring(acuity)}.")
    return {k: " ".join(v) for k, v in response_dict.items()}

def stringify_codes(codes):
    """ takes in list of dictionaries """
    def _stringify_icd(code_type, code_value, code_text):
        return f"ICD-{str(code_type)} code: [{str(code_value).strip().replace('.', '').lower()}], {str(code_text).strip().lower()}."
    
    return_list = [f"The patient received the following diagnostic codes:"]
    for code in codes:
        return_list.append(_stringify_icd(**code))
    return {"codes": " ".join(return_list)}

def stringify_pyxis(meds):
    """ takes in list of dictionaries """
    def _stringify_pyxis(charttime, med_list):
        return f"At {charttime}, {', '.join([str(s).lower() for s in med_list])} were administered."
    
    return_list = [f"The patient received the following medications:"]
    for timestamp in meds:
        return_list.append(_stringify_pyxis(**timestamp))
    return {"pyxis": " ".join(return_list)}

def stringify_vitals(vitals):
    """ takes in list of dictionaries """
    def _stringify_vitals(charttime, temperature, heartrate, resprate, o2sat, sbp, dbp, pain):
        procstring = lambda s: str(s).lower() if s is not None and str(s) != 'nan' else "not recorded"
        response_list = [f"At {charttime},"]
        response_list.append(f"temperature was {procstring(temperature)},")
        response_list.append(f"pulse was {procstring(heartrate)},")
        response_list.append(f"respirations was {procstring(resprate)},")
        response_list.append(f"o2 saturation was {procstring(o2sat)},")
        response_list.append(f"systolic blood pressure was {procstring(sbp)},")
        response_list.append(f"diastolic blood pressure was {procstring(dbp)},")
        response_list.append(f"pain was {procstring(pain)}.")
        return " ".join(response_list)
    
    return_list = [f"The patient had the following vitals:"]
    for timestamp in vitals:
        return_list.append(_stringify_vitals(**timestamp))
    return {"vitals": " ".join(return_list)}

def stringify_medrecon(meds):
    """ takes in list of dictionaries """
    def _stringify_medrecon(med_name, med_description):
        return f"{str(med_name).lower()}, {str(med_description).lower()}."
    if len(meds) > 0:
        return_list = [f"The patient was previously taking the following medications:"]
        for med in meds:
            return_list.append(_stringify_medrecon(**med))
    else:
        return_list = [f"The patient was not on any previous medications."]
    return {"medrecon": " ".join(return_list)}
# %%
if __name__ == '__main__':

    # %%
    # ED STAY METAINFO
    print("Loading ED Stays")
    valid_dispositions = [
        'HOME', # 241632
        'ADMITTED', #158010
        # 'TRANSFER', # 7025
        # 'LEFT WITHOUT BEING SEEN', #6155
        # 'ELOPED', #5710
        # 'OTHER',  #4297
        # 'LEFT AGAINST MEDICAL ADVICE', #1881
        'EXPIRED' #377
    ]
    edstays = pd.read_csv(os.path.join(path, 'ed/edstays.csv.gz'), compression='gzip')
    edstays = edstays[edstays['disposition'].isin(valid_dispositions)]
    # external info
    # GET AGE
    ext_path = '/mnt/bigdata/compmed/physionet/mimic-iv-2.2/mimic-iv-2.2/'
    edstays = edstays.merge(
        pd.read_csv(os.path.join(ext_path, 'hosp/patients.csv.gz'), compression='gzip')\
            [['subject_id', 'anchor_age', 'dod']], 
            on='subject_id', how='left')
    # 
    # GET HOSPITAL INFORMATION
    edstays = edstays.merge(
        pd.read_csv(os.path.join(ext_path, 'hosp/admissions.csv.gz'), compression='gzip')\
            [['subject_id', 'hadm_id', 'admittime', 'dischtime', 'discharge_location', 'insurance', 'language', 'marital_status', 'hospital_expire_flag']],
        on=['subject_id', 'hadm_id'], how='left')
    
    # %%
    print("processing ED Stays")
    edstays_string_dict = {}
    for _, r in tqdm(edstays.iterrows()):
        edstays_dict = dict(
            subject_id = r.get('subject_id'),
            gender = r.get('gender'),
            race = r.get('race'),
            anchor_age = r.get('anchor_age'),
            arrival_transport = r.get('arrival_transport'),
            intime = r.get('intime'),
            outtime = r.get('outtime'),
            disposition = r.get('disposition'),
            hadm_id = r.get('hadm_id'),
            admittime = r.get('admittime'),
            dischtime = r.get('dischtime'),
            discharge_location = r.get('discharge_location'),
            insurance = r.get('insurance'),
            language = r.get('language'),
            marital_status = r.get('marital_status'),
            hospital_expire_flag = r.get('hospital_expire_flag ='),
            dod = r.get('dod'),
        )
        edstays_string_dict[r['stay_id']] = stringify_edstays(**edstays_dict)

    # %%
    # TRIAGE INFORMATION
    print('loading triage')
    triage = pd.read_csv(os.path.join(path, 'ed/triage.csv.gz'), compression='gzip')\
        .merge(edstays[['stay_id']], on='stay_id', how='inner')\
        .set_index('stay_id')
    # %% 
    print('processing triage')
    triage_string_dict = {}
    for idx, r in tqdm(triage.iterrows()):
        triage_dict = dict(
            temperature = r.get('temperature'), 
            heartrate = r.get('heartrate'), 
            resprate = r.get('resprate'),
            o2sat = r.get('o2sat'), 
            sbp = r.get('sbp'), 
            dbp = r.get('dbp'), 
            pain = r.get('pain'), 
            acuity = r.get('acuity'), 
            chiefcomplaint = r.get('chiefcomplaint')
        )
        triage_string_dict[idx] = stringify_triage(**triage_dict)

    # %%
    # PREVIOUS MEDICATIONS
    print('loading medrecon')
    medrecon = pd.read_csv(os.path.join(path, 'ed/medrecon.csv.gz'), compression='gzip')\
        .merge(edstays[['stay_id']], on='stay_id', how='inner')\
        .set_index('stay_id')\
        .sort_values(by='charttime')
    # %%
    print('processing medrecon')
    # will need a groupby
    medrecon_string_dict = {}
    for idx in tqdm(medrecon.index.unique()):
        if type(medrecon.loc[idx]) == pd.Series:
            r = medrecon.loc[idx]
            medrecon_list = [dict(med_name=r.get('name'), med_description=r.get('etcdescription'))]
        else:
            medrecon_list = [dict(med_name=r.get('name'), med_description=r.get('etcdescription')) for _, r in medrecon.loc[idx].iterrows()]
        medrecon_string_dict[idx] = stringify_medrecon(medrecon_list)


    # %%
    # MEDICATIONS ADMINISTERED IN ER
    print('loading pyxis')
    pyxis = pd.read_csv(os.path.join(path, 'ed/pyxis.csv.gz'), compression='gzip')\
        .merge(edstays[['stay_id']], on='stay_id', how='inner')\
        .set_index('stay_id')\
        .sort_values(by='charttime')
    # %%
    # charttime, med_list
    print('processing pyxis')
    # TODO: drop duplicates?
    pyxis_string_dict = {}
    for idx in tqdm(pyxis.index.unique()):
        if type(pyxis.loc[idx]) == pd.Series:
            r = pyxis.loc[idx]
            pyxis_list = [dict(charttime=r.get('charttime'), med_list=r.get('name'))]
        else:
            tdf = pyxis.loc[idx].groupby('charttime')['name'].apply(lambda ll: list(ll)).reset_index()
            pyxis_list = [dict(charttime=r.get('charttime'), med_list=r.get('name')) for _, r in tdf.iterrows()]
        pyxis_string_dict[idx] = stringify_pyxis(pyxis_list)
    
    # %%
    # DISCHARGE CODES FROM ER ONLY
    print('loading diagnosis')
    diagnosis = pd.read_csv(os.path.join(path, 'ed/diagnosis.csv.gz'), compression='gzip')\
        .merge(edstays[['stay_id']], on='stay_id', how='inner')\
        .set_index('stay_id')\
        .sort_values(by='seq_num')
    # %%
    # will need a groupby
    print('processing diagnosis')
    diagnosis_string_dict = {}
    for idx in tqdm(diagnosis.index.unique()):
        if type(diagnosis.loc[idx]) == pd.Series:
            r = diagnosis.loc[idx]
            diagnosis_list = [dict(code_type=r.get('icd_version'), code_value=r.get('icd_code'), code_text=r.get('icd_title'))]
        else:
            diagnosis_list = [dict(code_type=r.get('icd_version'), code_value=r.get('icd_code'), code_text=r.get('icd_title')) for _, r in diagnosis.loc[idx].iterrows()]
        diagnosis_string_dict[idx] = stringify_codes(diagnosis_list)

    # %%
    # MEASURED VITAL SIGNS
    print('loading vitals')
    vitalsign = pd.read_csv(os.path.join(path, 'ed/vitalsign.csv.gz'), compression='gzip')\
        .merge(edstays[['stay_id']], on='stay_id', how='inner')\
        .set_index('stay_id')\
        .sort_values(by='charttime')

    # %%
    print('processing vitals')
    vitalsign_string_dict = {}
    for idx in tqdm(vitalsign.index.unique()):
        if type(vitalsign.loc[idx]) == pd.Series:
            r = vitalsign.loc[idx]
            vitalsign_list = [dict(
                charttime=r.get('charttime'), 
                temperature=r.get('temperature'), 
                heartrate=r.get('heartrate'),
                resprate=r.get('resprate'),
                o2sat=r.get('o2sat'),
                sbp=r.get('sbp'),
                dbp=r.get('dbp'),
                pain=r.get('pain')
                )]
        else:
            vitalsign_list = [dict(
                charttime=r.get('charttime'), 
                temperature=r.get('temperature'), 
                heartrate=r.get('heartrate'),
                resprate=r.get('resprate'),
                o2sat=r.get('o2sat'),
                sbp=r.get('sbp'),
                dbp=r.get('dbp'),
                pain=r.get('pain')
                
                ) for _, r in vitalsign.loc[idx].iterrows()]
        vitalsign_string_dict[idx] = stringify_vitals(vitalsign_list)



    # %%
    print('merging...')
    final_dict = edstays_string_dict 
    print('adding triage')
    for k, v in triage_string_dict.items():
        final_dict[k].update(v)
    print('adding medrecon')
    for k, v in medrecon_string_dict.items():
        final_dict[k].update(v)
    print('adding vitalsigns')
    for k, v in vitalsign_string_dict.items():
        final_dict[k].update(v)
    print('adding pyxis')
    for k, v in pyxis_string_dict.items():
        final_dict[k].update(v)
    print('adding diagnosis')
    for k, v in diagnosis_string_dict.items():
        final_dict[k].update(v)
    print('saving...')
    with open(os.path.join(path, "text_repr.json"), 'w') as f:
        json.dump(final_dict, f)

    print('done.')
    # %%
    # HOME                            155423
    # HOME HEALTH CARE                 75572
    # SKILLED NURSING FACILITY         43024
    # REHAB                            10523
    # DIED                              8511
    # CHRONIC/LONG TERM ACUTE CARE      7144
    # HOSPICE                           3469
    # AGAINST ADVICE                    2590
    # PSYCH FACILITY                    2262
    # ACUTE HOSPITAL                    1610
    # OTHER FACILITY                    1355
    # ASSISTED LIVING                    551
    # HEALTHCARE FACILITY                 42
    # like ICU status

    # %%