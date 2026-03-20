import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df = pd.read_csv('/work/pi_dagarwal_umass_edu/project_7/hmagapu/ordered_song_list.csv')

df_sorted = df.sort_values('play_count', ascending=False).reset_index(drop=True)
total_plays = df['play_count'].sum()
df_sorted['cum_pct'] = df_sorted['play_count'].cumsum() / total_plays * 100
idx_80 = (df_sorted['cum_pct'] >= 80).idxmax()

# Plot 1: Histogram + Frequency Distribution
df_cap = df[df['play_count'] <= 200]

fig, ax1 = plt.subplots(figsize=(12, 8))

n, bins, _ = ax1.hist(df_cap['play_count'], bins=100, label='Track Count')
ax1.set_yscale('log')
ax1.set_xlabel('Play Count per Track', fontsize=12)
ax1.set_ylabel('Number of Tracks (log)', fontsize=12)

ax2 = ax1.twinx()
bin_centers = 0.5 * (bins[:-1] + bins[1:])
rel_freq = n / n.sum() * 100
ax2.plot(bin_centers, rel_freq, linewidth=2, color='#ff6b6b', marker='o', markersize=3, label='Relative Freq %')
ax2.set_ylabel('Relative Frequency (%)', fontsize=12)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
ax1.set_title('Distribution of Song Play Counts (LastFM 1K)', fontsize=13, pad=12)

plt.tight_layout()
plt.savefig('/work/pi_dagarwal_umass_edu/project_7/srikar/dolby-research/histogram_v2.png', dpi=150)
plt.show()

# Plot 2: Pareto Curve + Frequency Distribution
fig, ax1 = plt.subplots(figsize=(12, 8))

ax1.plot(df_sorted.index + 1, df_sorted['cum_pct'], linewidth=1.8, label='Cumulative Plays %')
ax1.fill_between(df_sorted.index + 1, df_sorted['cum_pct'], alpha=0.15)
ax1.axhline(80, linestyle='--', linewidth=1.2, label='80% threshold')
ax1.axvline(idx_80 + 1, linestyle=':', linewidth=1.2, label=f'~{idx_80+1:,} tracks (80%)')
ax1.set_xlabel('Track Rank (sorted by play count)', fontsize=12)
ax1.set_ylabel('Cumulative % of Total Plays', fontsize=12)
ax1.xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))

ax2 = ax1.twinx()
bucket = 5000
play_counts_sorted = df_sorted['play_count'].values
rank_bins = np.arange(0, len(df_sorted) + bucket, bucket)
freq_vals, bin_mids = [], []
for i in range(len(rank_bins) - 1):
    chunk = play_counts_sorted[rank_bins[i]:rank_bins[i+1]]
    freq_vals.append(chunk.mean() if len(chunk) > 0 else 0)
    bin_mids.append((rank_bins[i] + rank_bins[i+1]) / 2)

ax2.bar(bin_mids, freq_vals, width=bucket * 0.9, alpha=0.35, label='Avg Play Count / Bucket')
ax2.set_ylabel('Avg Play Count per 5k-Track Bucket', fontsize=11)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)
ax1.set_title('Pareto Curve: Play Concentration (LastFM 1K)', fontsize=13, pad=12)

plt.tight_layout()
plt.savefig('/work/pi_dagarwal_umass_edu/project_7/srikar/dolby-research/histogram112.png', dpi=150)