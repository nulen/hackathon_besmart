# -*- coding: utf-8 -*-
#
# besmart.energy
#
# Hackathon script
#
# Copyright 2019 Atende Software
#

import datetime
from octopus_client import *


# Dates are in UTC format
date_since = int(round(datetime.datetime(2016, 1, 1, 0, 0).timestamp() * 1000))
date_till = int(round(datetime.datetime(2019, 5, 1, 0, 0).timestamp() * 1000))

# Creating besmart octopus API client
oc = OctopusClient(auth='', address='hackaton.besmart.energy')

# Fetching response
result = oc.send_single_request(TStorageGet(
    cid=1338,
    mid=4,
    type='consumption',
    since=date_since,
    till=date_till,
    delta_t=60
))

# Converting response into pandas DataFrame
df = oc.get_df_from_result(result)

# Saving results into csv (index, value)
df.to_csv('hackathon_data.csv', ',')
