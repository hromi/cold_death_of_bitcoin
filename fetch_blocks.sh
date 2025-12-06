#!/bin/bash

# Create a directory for the data
mkdir -p btc_blocks
cd btc_blocks

# Download and extract data
start_date="20250101"
end_date="20251205"
current_date="$start_date"

while [ "$current_date" -le "$end_date" ]; do
    FILE_URL="https://gz.blockchair.com/bitcoin/blocks/blockchair_bitcoin_blocks_${current_date}.tsv.gz"
    wget -q $FILE_URL
    gunzip "blockchair_bitcoin_blocks_${current_date}.tsv.gz"
    current_date=$(date -I -d "$current_date + 1 day" | tr -d '-')
done

# Extract block ID (column 1) and time (column 3), skip header
echo -e "id\ttime" > combined_times.tsv

for FILE in blockchair_bitcoin_blocks_*.tsv; do
    tail -n +2 "$FILE" | awk -F'\t' '{print $1 "\t" $3}' >> combined_times.tsv
done
