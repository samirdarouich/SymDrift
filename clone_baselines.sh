#!/bin/bash

mkdir -p baselines

repos=(
  "https://github.com/MinkaiXu/GeoDiff"
  "https://github.com/PattanaikL/GeoMol"
  "https://github.com/apple/ml-mcf"
  "https://github.com/gcorso/torsional-diffusion"
  "https://github.com/shenoynikhil/ETFlow"
)

for repo in "${repos[@]}"; do
  dest="baselines/$(basename "$repo")"
  git clone "$repo" "$dest"
  rm -rf "$dest/.git"
done
