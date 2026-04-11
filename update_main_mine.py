import re
import os

filepath = 'main_mine.py'
with open(filepath, 'r') as f:
    content = f.read()

# 1. Scan for existing work logic
scan_logic = """
    # RESUME LOGIC: Scan for existing experts and reserves
    existing_experts = [f for f in os.listdir("storage/models") if f.startswith("strat_") and f.endswith(".txt")]
    existing_reserves = [f for f in os.listdir("storage/reserve") if f.startswith("reserve_") and f.endswith(".txt")]

    strategy_id = len(existing_experts)
    reserve_count = len(existing_reserves)

    portfolio = []
    if os.path.exists("storage/active_portfolio/league_summary.json"):
        with open("storage/active_portfolio/league_summary.json", "r") as f:
            portfolio = json.load(f)
            # Only keep active experts that actually have a model file
            portfolio = [s for s in portfolio if os.path.exists(f"storage/models/strat_{portfolio.index(s)}.txt")]
            # Wait, the index might not match if some were deleted.
            # Better to rely on the ID in the JSON if we want to be robust,
            # but for now strat_N.txt is standard.

    print(f"📊 Resuming: {len(portfolio)} experts in portfolio, {strategy_id} total expert files, {reserve_count} reserve files.")
"""

# Replace the initialization of portfolio, strategy_id and the directory creation with more robust logic
# Find the line "os.makedirs("storage/reserve", exist_ok=True)" and insert after it.

old_init = """    portfolio = []
    strategy_id = 0
    os.makedirs("storage/models", exist_ok=True)
    os.makedirs("storage/active_portfolio", exist_ok=True)
    os.makedirs("storage/reserve", exist_ok=True)"""

new_init = """    os.makedirs("storage/models", exist_ok=True)
    os.makedirs("storage/active_portfolio", exist_ok=True)
    os.makedirs("storage/reserve", exist_ok=True)

    # RESUME LOGIC: Scan for existing experts and reserves
    existing_experts = [f for f in os.listdir("storage/models") if f.startswith("strat_") and f.endswith(".txt")]
    existing_reserves = [f for f in os.listdir("storage/reserve") if f.startswith("reserve_") and f.endswith(".txt")]

    strategy_id = len(existing_experts)
    reserve_count = len(existing_reserves)

    portfolio = []
    if os.path.exists("storage/active_portfolio/league_summary.json"):
        with open("storage/active_portfolio/league_summary.json", "r") as f:
            portfolio = json.load(f)
            # Keep what we have files for
            portfolio = portfolio[:strategy_id]

    print(f"📊 Resuming: {len(portfolio)} experts in portfolio, {strategy_id} total expert files, {reserve_count} reserve files.")"""

content = content.replace(old_init, new_init)

# Also need to make sure the loops start correctly.
# Expert loop: while len(portfolio) < config['portfolio_size']:
# This is already fine because len(portfolio) is resumed.

# Reserve loop initialization: reserve_count = 0
# Need to replace it so it uses the scanned reserve_count.
content = content.replace("    reserve_count = 0", "    # reserve_count is already set by resume logic")

with open(filepath, 'w') as f:
    f.write(content)
