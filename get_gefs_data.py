from tqdm import tqdm
import datetime
import boto3, io, os
from botocore import UNSIGNED
from botocore.client import Config
from nimbl.utils import s3_partial_grib2_get_responses, s3_responses_to_tempfiles


#copied from Adam Schnapp /scratch1/NCEPDEV/mdl/Adam.Schnapp/noscrub/bc_test/get_gfs_10mTMP.py
s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

bucket = 'noaa-gefs-pds'
#match = '(:HGT:surface:|:TMP:2 m above ground:|:DPT:2 m above ground:|:10 m above ground:)'
#match = '(:APCP:surface:|:PWAT:entire atmosphere (considered as a single layer):)'
match = ':APCP:surface:'
days_ago=0
lead_time = '120'
hh = '12'
for member in tqdm(range(1,35)):
#for days_ago in [0]:#range(50):
    first_day = datetime.date(2023,10,1)
    this_day = first_day - datetime.timedelta(days=days_ago)
    mem=f'{member:02d}'
    #https://noaa-gefs-pds.s3.amazonaws.com/gefs.20210601/12/atmos/pgrb2ap5/gec00.t12z.pgrb2a.0p50.f000
    key = f'gefs.{this_day.strftime("%Y%m%d")}/{hh}/atmos/pgrb2ap5/gep{mem}.t{hh}z.pgrb2a.0p50.f{lead_time}'
    
    responses = s3_partial_grib2_get_responses(s3_client, bucket, key, match)
    tempfiles = s3_responses_to_tempfiles(responses)
    gefs_dir = f"/scratch2/STI/mdl-sti/Sidney.Lower/test_data/gefs/{this_day.strftime('%Y%m%d')}/"
    if not os.path.isdir(gefs_dir):
        os.mkdir(gefs_dir)
    outfile = f"/scratch2/STI/mdl-sti/Sidney.Lower/test_data/gefs/{this_day.strftime('%Y%m%d')}/gefs{mem}.t{hh}z.f{lead_time}"
    with io.open(outfile, mode='bw') as f:
        for tfile in tempfiles:
            f.write(tfile.read())


