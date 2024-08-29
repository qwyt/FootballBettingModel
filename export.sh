#!/bin/bash

notebook_files=("workbench/EDA.ipynb" "workbench/TabularModel.ipynb")
notebook_files=()


for file in "${notebook_files[@]}"; do
  base_name=$(basename "${file%.ipynb}")

  SUPPRESS_WARNINGS=1 jupyter nbconvert "$file" --HTMLExporter.extra_template_basedirs=/home/paulius/data/projects/football_m2_s4/workbench/utils/theme --HTMLExporter.template_file=fivethirtyeight --to html --output "$base_name" --no-input --output-dir='./output'
#  SUPPRESS_WARNINGS=1 jupyter nbconvert "$file" --to html --output "$base_name" --no-input --output-dir='./output'
#  SUPPRESS_WARNINGS=1 jupyter nbconvert --to html --output "$base_name" --no-input --output-dir='./output' "$file"
#  SUPPRESS_WARNINGS=1 jupyter nbconvert --to html --template fivethirtyeight --theme dark --output "$base_name" --no-input --output-dir='./output' "$file"
#  SUPPRESS_WARNINGS=1 jupyter nbconvert --to html --template fivethirtyeight --theme dark --output "${base_name}_with_code" --output-dir='./output' "$file"
#  SUPPRESS_WARNINGS=1 jupyter nbconvert --to html --output "${base_name}_with_code" --output-dir='./output' "$file"
done

if [ "$1" == "publish" ]; then
  cd output
  git add .
  git commit -m "Publish"
  git push
fi
#F0F0F0