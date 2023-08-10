# %% 
import os 
import pandas as pd 
import json 
from tqdm import tqdm

from utils import stringify_visit_codes, stringify_visit_labs, stringify_visit_meds, stringify_visit_meta

# %%

def convert_admissions(path, output='admissions_text.json'):
    admissions = pd.read_csv(os.path.join(path, 'hosp/admissions.csv.gz'), compression='gzip')

    res = {}

    for _, r in tqdm(admissions.iterrows()):
        res[r['hadm_id']] = stringify_visit_meta(
            subject_id = r.get('subject_id'),
            hadm_id = r.get('hadm_id'),
            admittime=r.get('admittime'),
            dischtime=r.get('dischtime'),
            deathtime=r.get('deathtime'),
            admission_type=r.get('admission_type'),
            admission_location=r.get('admission_location'),
            discharge_location=r.get('discharge_location'),
            insurance=r.get('insurance'), 
            language=r.get('language'),
            marital_status=r.get('marital_status'), 
            race=r.get('race')
        )

    with open(os.path.join(path, output), "w") as f:
        json.dump(dict(res), f)
    return os.path.join(path, output)

# %%

# work domain by domain to be more flexible later
def convert_codes(path, output = "codes_text.json"):
    codes = pd.read_csv(os.path.join(path, 'hosp/diagnoses_icd.csv.gz'), compression='gzip')
    code_key = pd.read_csv(os.path.join(path, 'hosp/d_icd_diagnoses.csv.gz'), compression='gzip').set_index(['icd_code', 'icd_version'])
    
    def _stringify(tdf):
        # tdf = codes[codes.hadm_id==22580999] # .sort_values(by='seq_num')
        this_codes = [dict(
                code_type = r['icd_version'],
                code_value = r['icd_code'],
                code_text=code_key.loc[r['icd_code'], int(r['icd_version'])].long_title
            ) for _, r in tdf.iterrows()]
        this_string = stringify_visit_codes(this_codes)
        return this_string
    
    res = codes.groupby('hadm_id').apply(_stringify)
    with open(os.path.join(path, output), "w") as f:
        json.dump(dict(res), f)
    return os.path.join(path, output)

# %%
def convert_labs(path, output = "labs_text.json"):

    labs = pd.read_csv(os.path.join(path, 'hosp/labevents.csv.gz'), compression='gzip')
    labs_key = pd.read_csv(os.path.join(path, 'hosp/d_labitems.csv.gz'), compression='gzip').set_index(['itemid'])
    def _stringify(tdf):
        # tdf = codes[codes.hadm_id==22580999] # .sort_values(by='seq_num')
        this_labs = [dict(
                valueuom = r.get('valueuom'),
                lab_value = r.get('value'),
                lab_name=labs_key.loc[r['itemid']].label,
                flag=r.get('flag'),
                datetime=r.get('storetime')
            ) for _, r in tdf.iterrows()]
        this_string = stringify_visit_labs(this_labs)
        return this_string

    res = labs.groupby('hadm_id').apply(_stringify)
    with open(os.path.join(path, output), "w") as f:
        json.dump(dict(res), f)
    return os.path.join(path, output)

def convert_meds(path, output = "meds_text.json"):

    meds = pd.read_csv(os.path.join(path, 'hosp/prescriptions.csv.gz'), compression='gzip')
    
    def _stringify(tdf):
        # tdf = codes[codes.hadm_id==22580999] # .sort_values(by='seq_num')
        this_meds = [dict(
                drug = r.get('drug'),
                route = r.get('route'),
                starttime=r.get('starttime'),
                endtime=r.get('endtime')
            ) for _, r in tdf.iterrows()]
        this_string = stringify_visit_meds(this_meds)
        return this_string

    res = meds.groupby('hadm_id').apply(_stringify)
    with open(os.path.join(path, output), "w") as f:
        json.dump(dict(res), f)
    return os.path.join(path, output)

# %%
if __name__ == '__main__':
    path = '/mnt/bigdata/compmed/physionet/mimic-iv-clinical-database-demo-2.2/mimic-iv-clinical-database-demo-2.2/'
    codes_p = convert_codes(path)
    labs_p = convert_labs(path)
    meds_p = convert_meds(path)
    adm_p = convert_admissions(path)
    # test_df = pd.read_csv(os.path.join(path, "hosp/admissions.csv.gz"), compression='gzip')
    # To test:
    # with open(adm_p, 'r') as f:
    #     admissions = json.load(f)

    # admissions