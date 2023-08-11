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
    labs_key = pd.read_csv(os.path.join(path, 'hosp/d_labitems.csv.gz'), compression='gzip').set_index(['itemid'])
    def _dataprep(r):
        return dict(
            valueuom = r.get('valueuom'),
            lab_value = r.get('value'),
            lab_name=labs_key.loc[r['itemid']].label,
            flag=r.get('flag'),
            datetime=r.get('storetime'))
    
    # NOTE: the data are too big, need to process in chunks.
    print('processing in chunks...')
    labs_chunked = pd.read_csv(os.path.join(path, 'hosp/labevents.csv.gz'), compression='gzip', chunksize=50000)
    data_to_convert = {}
    for labs in tqdm(labs_chunked):
        # filter where hadm_id is not null
        labs = labs[~labs.hadm_id.isna() & ~labs['value'].isna()] # only labs with values
        for _, r in labs.iterrows():
            if r['hadm_id'] not in data_to_convert:
                data_to_convert[r['hadm_id']] = []
            data_to_convert[r['hadm_id']].append(_dataprep(r))
    print('prepped.')
    # NOTE: This takes too much memory. Will replace with a manual way.
    # def _stringify(tdf):
    #     # tdf = codes[codes.hadm_id==22580999] # .sort_values(by='seq_num')
    #     this_labs = [dict(
    #             valueuom = r.get('valueuom'),
    #             lab_value = r.get('value'),
    #             lab_name=labs_key.loc[r['itemid']].label,
    #             flag=r.get('flag'),
    #             datetime=r.get('storetime')
    #         ) for _, r in tdf.iterrows()]
    #     this_string = stringify_visit_labs(this_labs)
    #     return this_string
    # res = labs.groupby('hadm_id').apply(_stringify)
    print('stringifying...')
    res = {k: stringify_visit_labs(v) for k, v in tqdm(data_to_convert.items())}
    print('writing...')
    with open(os.path.join(path, output), "w") as f:
        json.dump(dict(res), f)
    print('done.')
    return os.path.join(path, output)

def convert_meds(path, output = "meds_text.json"):
    print('loading data...')
    meds = pd.read_csv(os.path.join(path, 'hosp/prescriptions.csv.gz'), compression='gzip')
    meds = meds[~meds.hadm_id.isna()]
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
    print('processing...')
    res = meds.groupby('hadm_id').apply(_stringify)
    with open(os.path.join(path, output), "w") as f:
        json.dump(dict(res), f)
    print('done.')
    return os.path.join(path, output)

# %%
if __name__ == '__main__':
    # JC: I ran these on my local machine since i had a copy of the repo...
    path = '/mnt/bigdata/compmed/physionet/mimic-iv-clinical-database-demo-2.2/mimic-iv-clinical-database-demo-2.2/'
    # path = '/mnt/bigdata/compmed/physionet/mimic-iv-2.2/mimic-iv-2.2/'
    print("Running codes")
    codes_p = convert_codes(path)
    print("Codes complete.")
    print("Running meds")
    meds_p = convert_meds(path)
    print("Meds complete.")
    print("Running labs")
    labs_p = convert_labs(path)
    print("Labs complete.")
    print("Running admissions.")
    adm_p = convert_admissions(path)
    print("Done.")
    # test_df = pd.read_csv(os.path.join(path, "hosp/admissions.csv.gz"), compression='gzip')
    # To test:
    # with open(adm_p, 'r') as f:
    #     admissions = json.load(f)

    # admissions
