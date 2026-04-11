import re

filepath = 'main_mine.py'
with open(filepath, 'r') as f:
    lines = f.readlines()

# Identify the save block
save_block = [
    '    portfolio_df = pd.DataFrame(portfolio)\n',
    '    portfolio_df.to_pickle("storage/active_portfolio/full_steam_30.pkl")\n',
    '    with open("storage/active_portfolio/league_summary.json", "w") as f:\n',
    '        json.dump(portfolio, f, indent=4)\n'
]

# Find where to insert it in the expert loop
# After strategy_id += 1 and the print(f"✅ HIRED! ...")

new_lines = []
skip_next_save_block = False

for line in lines:
    new_lines.append(line)
    if 'print(f"✅ HIRED! EV: {best_ev:.3f} | ATR: {atr_mult:.2f} | RRR: {rrr:.2f} | MC: {limit:.1f}R")' in line:
        # Add the save block here with proper indentation (16 spaces)
        new_lines.append('                # Incremental Save\n')
        new_lines.append('                portfolio_df = pd.DataFrame(portfolio)\n')
        new_lines.append('                portfolio_df.to_pickle("storage/active_portfolio/full_steam_30.pkl")\n')
        new_lines.append('                with open("storage/active_portfolio/league_summary.json", "w") as f:\n')
        new_lines.append('                    json.dump(portfolio, f, indent=4)\n')

# Now we need to remove the original save block which is outside the loop
# It usually comes before # RESERVE MINING

final_lines = []
i = 0
while i < len(new_lines):
    if 'portfolio_df = pd.DataFrame(portfolio)' in new_lines[i] and '    ' == new_lines[i][:4] and '        ' != new_lines[i][:8]:
        # This is the outer one (4 spaces indent)
        # Check if next lines match the save block
        if i+3 < len(new_lines) and 'json.dump(portfolio, f, indent=4)' in new_lines[i+3]:
            i += 4 # Skip the 4 lines of the block
            continue
    final_lines.append(new_lines[i])
    i += 1

with open(filepath, 'w') as f:
    f.writelines(final_lines)
