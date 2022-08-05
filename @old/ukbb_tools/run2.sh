#!/bin/bash

case=$(ls ~/HSAM/hsam/)

for i in $case
do
python ~/tvb-pipeline/tvb-ukbb/bb_pipeline_tools/bb_pipeline.py "$i"
echo "$i" is done
done

