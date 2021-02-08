# -*- coding:utf-8 -*-

import json
import argparse
from HRec import pipeline

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str)

args = parser.parse_args()
model_name = args.model
print(args.model)
config = json.load(open("./configs/%s.json" % (model_name)))

p = pipeline.HProcess(config)
p.fit()
