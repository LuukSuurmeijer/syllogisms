#!/bin/bash

# This file is the pipeline generates a full DFS space + sentences and a subset-reduced Space + sentences

# 1: Original world file
# 2: Original sentence file
# 3: Original dimensionality
# 4: threads
# 5: Reduced world file
# 6: Reduced sent file
# 7: Reduced dimensionality

# Generate the full world + sentences

swipl -q -f syllogism_space.pl -g "gen_data('$2', '$1', $3, $4), halt."

# Reduce dimensionality of full world

python dfs_msubset.py $1 $5 $7 5000

# Load into dfs-tool, regenerate sentences
# This throws an error atm but it still works?
swipl -q -f syllogism_space.pl -g "gen_data('$5', '$6'), halt."
