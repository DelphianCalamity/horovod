#!/bin/bash

for x in `find * -type f -name 'values.csv'`
do
  echo `dirname $x`
    python double_exponential_plot.py --path=`dirname $x`
  done
