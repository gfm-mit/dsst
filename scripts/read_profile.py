import pstats

stats = pstats.Stats('output_file.prof')
stats.sort_stats('cumulative')
stats.print_stats(30)  # Print the top 10 functions by cumulative time