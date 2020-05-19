#!/bin/bash

for x in `find * -type f -name '*.csv'`
do
	echo dirname $x
	python plot.py --path=`dirname $x`
done
