# Search for a specific triplet
python total_correlation_value_key_finder.py -o yeast -t H3K4me1 H3K4me3 H3K27me3

# Search for all triplets containing H3K4me1
python total_correlation_value_key_finder.py -o yeast -s H3K4me1

# Get top 5 highest correlations
python total_correlation_value_key_finder.py -o yeast --top 5

# Get top 5 lowest correlations
python total_correlation_value_key_finder.py -o yeast --top 5 --lowest

# Get correlations in a specific range
python total_correlation_value_key_finder.py -o yeast --range 1.0 2.0

# Show statistics
python total_correlation_value_key_finder.py -o yeast --stats

# Export results to file
python total_correlation_value_key_finder.py -o yeast --top 10 --export top_10_results.json