#!/bin/bash
mkdir -p release && cmake ../.. -DCMAKE_BUILD_TYPE=Release $@ && cd ..
mkdir -p debug && cmake ../.. -DCMAKE_BUILD_TYPE=Debug $@ && cd ..
