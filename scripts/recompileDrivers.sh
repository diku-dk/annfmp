#!/bin/bash
cd annfmp/openclirreg/src/futhark
futhark pkg sync
futhark cuda driverKNN.fut --library
cd ..
cd ..
cd ..
cd ..
