"""
notebook_cells.py
=================
Copy-paste reference for simulator.ipynb cells.

This file contains the EXACT code to paste into new notebook cells.
Each section below is one cell — separated by clearly marked headers.

HOW TO USE:
-----------
1. Open simulator.ipynb
2. Find the cell that generates `alpha_list` (the list comprehension with
   `ace.generate_alpha(...)` — currently around execution_count 17)
3. **ADD** Cell A right after that cell
4. **REPLACE** the main simulation loop cell (the one starting with
   `import os` / `import time` / `batch_size = 200`) with Cell B

That's it. The DB file `simulation_history.db` is created automatically.
"""

# ============================================================================
# CELL A: Initialize session history + filter already-simulated alphas
# ============================================================================
# Paste this into a NEW cell right after the alpha_list generation cell.
# ============================================================================

CELL_A = """
from session_history import SessionHistoryDB

# Initialize the session history database
db = SessionHistoryDB("simulation_history.db")

# Show what's already been simulated for this region/universe
summary = db.get_session_summary(region, universe)
print(f"📊 Session History for {region}/{universe}:")
print(f"   Total recorded: {summary['total']}")
print(f"   Successful:     {summary['success']}")
print(f"   Errors:         {summary['error']}")

# Filter out already-simulated alphas
original_count = len(alpha_list)
alpha_list = db.filter_unsimulated(alpha_list, region, universe)

print(f"\\n🔍 Batch analysis:")
print(f"   Original batch size: {original_count}")
print(f"   Already simulated:   {original_count - len(alpha_list)}")
print(f"   Remaining to run:    {len(alpha_list)}")

if not alpha_list:
    print("\\n✅ All alphas in this batch have already been simulated!")
    print("   Change your parameters or expressions to generate new alphas.")
"""

# ============================================================================
# CELL B: Updated simulation loop with automatic DB recording
# ============================================================================
# Paste this to REPLACE the existing simulation loop cell.
# ============================================================================

CELL_B = """
import os
import time
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing. Cannot send message.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Failed to send telegram message: {e}")

def wait_for_telegram_command(expected_command="/continue"):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    offset = None
    try:
        res = requests.get(url, timeout=10).json()
        if res.get("ok") and len(res["result"]) > 0:
            offset = res["result"][-1]["update_id"] + 1
    except Exception:
        pass

    prompt = f"Paused. Reply with '{expected_command}' in Telegram to request a new link and continue."
    print(prompt)
    send_telegram_message(prompt)
    
    while True:
        try:
            req_url = f"{url}?timeout=30"
            if offset:
                req_url += f"&offset={offset}"
            res = requests.get(req_url, timeout=35).json()
            if res.get("ok"):
                for update in res["result"]:
                    offset = update["update_id"] + 1
                    if "message" in update and "text" in update["message"]:
                        text = update["message"]["text"].strip()
                        if text == expected_command:
                            send_telegram_message(f"Command '{expected_command}' received. Resuming script...")
                            return
        except requests.exceptions.Timeout:
            continue
        except Exception as e:
            print(f"Error checking Telegram updates: {e}")
            time.sleep(5)

# --- Main simulation loop with session history ---

batch_size = 200
all_simulation_results = []
total_simulated = 0

remaining_msg = f"🚀 Simulation started. Alphas to process: {len(alpha_list)}"
print(remaining_msg)
send_telegram_message(remaining_msg)

while alpha_list:
    batch_alphas = alpha_list[:batch_size]
    
    print(f"Simulating next {len(batch_alphas)} alphas...")
    
    result = ace.simulate_alpha_list_multi(s, batch_alphas)
    
    all_simulation_results.append(result)
    
    # ---- Record results to session history DB ----
    db.record_batch_results(batch_alphas, result, region, universe)
    
    total_simulated += len(batch_alphas)
    
    del alpha_list[:batch_size]
    
    summary = db.get_session_summary(region, universe)
    status_msg = (
        f"✅ Batch complete. {total_simulated} simulated this session. "
        f"{len(alpha_list)} remaining.\\n"
        f"📊 DB totals — Success: {summary['success']}, "
        f"Error: {summary['error']}, Total: {summary['total']}"
    )
    print(status_msg)
    send_telegram_message(status_msg)
    
    if alpha_list:
        wait_for_telegram_command(expected_command="/link")
        ace.check_session_and_relogin(s)
        msg = "Session updated. Starting the next batch..."
        print(msg)
        send_telegram_message(msg)

final_msg = "🏁 Simulation process finished successfully."
print(final_msg)
send_telegram_message(final_msg)

final_summary = db.get_session_summary(region, universe)
print(f"\\n📊 Final DB Summary for {region}/{universe}:")
print(f"   Total:      {final_summary['total']}")
print(f"   Successful: {final_summary['success']}")
print(f"   Errors:     {final_summary['error']}")
"""

# ============================================================================
# CELL C (Optional): View session history
# ============================================================================

CELL_C = """
history_df = db.get_batch_history(region, universe)
print(f"Total records: {len(history_df)}")
history_df
"""
