#!/bin/bash

# Loop through every file matching the pattern "run*.sh"
for script in run_*.sh; do
    
    # Skip this script itself if it matches the pattern
    if [ "$script" == "run_everything.sh" ]; then continue; fi

    echo "=========================================="
    echo "Running: $script"
    echo "=========================================="
    
    # Execute the script
    bash "$script"
    
    # Optional: sleep for a few seconds between runs to let I/O settle
    sleep 2
done

echo "Done processing all scripts."