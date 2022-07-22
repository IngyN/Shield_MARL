#!/usr/bin/env bash

# ./compile_all_optimized.sh map_name number_of_agents

map=$1 # first cmd line arg : which map : simple Pentagon ISR MIT SUNY SUNYvar
n=$2 # second cmd line arg : number of agents 2-4

if [ $# -eq 3 ]
  then
    extra=$3
    programName="$map""_""$n""_agents_""$extra"
else
	programName="$map""_""$n""_agents"
fi

# to change dont forget to change in genshield_simple.py

{
 # try removing old files
{
rm shields/name.txt
} || {
 rm shields/"$programName".slugsin
}  || {
rm shields/"$programName".json
}
{
rm shields/"$programName".shield
} || {
rm shields/"$programName".structuredslugs

}

} || {
	 echo "rm failed"
}

echo "done removing old, generating structuredslugs"
echo "$programName" > shields/name.txt
# echo "$map" > shields/map.txt
# echo "$n" > shields/num_agents.txt

cd shields

if [ ! -d "grid_shields/$map" ]; then
  mkdir grid_shields/$map
fi

python3 gen_shield_grid.py $map $n $programName

# read number of shields
nshields=$(cat grid_shields/""$map""/shield_num.txt)
nshields=$(($nshields - 1))

cd ..

# echo $nshields

for value in $(seq 0 1 $nshields)
do
	# echo $value
	echo "\n--------------------------\nSynthesizing shield : ""$value""/""$nshields"
	echo "compiling into slugsin"
	python tools/StructuredSlugsParser/compiler.py shields/grid_shields/temp/"$programName""_""$value".structuredslugs > shields/grid_shields/temp/"$programName""_""$value".slugsin

	echo "computing strategies ..."

	./src/slugs --explicitStrategy --jsonOutput shields/grid_shields/temp/"$programName""_""$value".slugsin > shields/grid_shields/$map/"$programName""_""$value".json

	cd shields
	echo "adjusting slugs output ..."
	python Control_Parser.py grid_shields/$map/"$programName""_""$value"

	# echo "Sorting shield"
	# python3 sort_shield.py

	cd ..

	mv shields/grid_shields/$map/"$programName""_""$value".json shields/grid_shields/temp/"$programName""_""$value".json

done

echo "\n------------------------------\nDone"
