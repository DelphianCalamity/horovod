#!/bin/bash

for x in `find * -type f -name 'values.csv' -path $2'/*'`
do
  echo `dirname $x`
    python plot.py --path=`dirname $x` --model=$1 &
  done