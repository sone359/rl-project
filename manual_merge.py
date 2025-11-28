import os
import csv
import argparse
import sys

def merge_monitor_logs(old_path, new_path, save_path, start_step, end_step=None):
    """
    Merge two Monitor files (Stable Baselines 3).
    Format : 
      Line 1 : Metadata (JSON)
      Line 2 : Header (r, l, t)
      Lines 3+ : Data
    """
    if not os.path.exists(old_path):
        print(f"[ERROR] Missing Monitor file: {old_path}")
        return
    if not os.path.exists(new_path):
        print(f"[ERROR] Missing Monitor file: {new_path}")
        return

    print(f"[MONITOR] Merging '{old_path}' + '{new_path}' -> '{save_path}'")
    
    kept_lines = []
    cumulative_steps = 0
    
    # --- 1. Reading the OLD file ---
    with open(old_path, 'r') as f:
        # Line 1 : Metadata (Keep the one from the start of training)
        meta_line = f.readline()
        if meta_line: kept_lines.append(meta_line)
        
        # Line 2 : Header
        header_line = f.readline()
        if header_line: kept_lines.append(header_line)
        
        # Data
        reader = csv.DictReader(f, fieldnames=['r', 'l', 't'])
        for row in reader:
            try:
                l = int(row['l'])
            except (ValueError, TypeError):
                continue 
            
            # Keep as long as we are BEFORE the checkpoint
            if cumulative_steps + l <= start_step:
                kept_lines.append(f"{row['r']},{row['l']},{row['t']}\n")
                cumulative_steps += l
            else:
                # Stop as soon as we exceed the start_step
                break

    print(f"  -> Old data cut at step {cumulative_steps} (Target start: {start_step})")

    # --- 2. Reading the NEW file ---
    new_lines_added = 0
    with open(new_path, 'r') as f:
        # Skip Metadata and Header of the new file
        f.readline()
        f.readline()
        
        reader = csv.DictReader(f, fieldnames=['r', 'l', 't'])
        for row in reader:
            try:
                l = int(row['l'])
            except (ValueError, TypeError):
                continue
            
            # Keep as long as we are WITHIN the desired end_step
            if end_step is None or (cumulative_steps + l <= end_step):
                kept_lines.append(f"{row['r']},{row['l']},{row['t']}\n")
                cumulative_steps += l
                new_lines_added += 1
            else:
                # Reached the desired end_step
                break
    
    print(f"  -> New data: {new_lines_added} episodes.")
    print(f"  -> Total Steps accumulated: {cumulative_steps}")

    # --- 3. Writing ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.writelines(kept_lines)
    print(f"  -> New data added: {new_lines_added} episodes.")
    print(f"  -> Saved to: {save_path}\n")

def merge_noise_logs(old_path, new_path, save_path, start_step, end_step=None):
    """
    Merge two Noise files (Custom CSV).
    Format :
      Line 1 : Header (timestep, ...)
      Lines 2+ : Data (column 0 = timestep)
    """
    if not os.path.exists(old_path):
        print(f"[ERROR] Missing Noise file: {old_path}")
        return
    if not os.path.exists(new_path):
        print(f"[ERROR] Missing Noise file: {new_path}")
        return

    print(f"[NOISE] Merging '{old_path}' + '{new_path}' -> '{save_path}'")

    kept_lines = []
    
    # --- 1. Reading the OLD file ---
    with open(old_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header:
            kept_lines.append(",".join(header) + "\n")
        
        for row in reader:
            if not row: continue
            try:
                ts = int(row[0]) # Suppose colonne 0 = timestep
                if ts <= start_step:
                    kept_lines.append(",".join(row) + "\n")
            except ValueError:
                continue
    
    print(f"  -> Old data kept up to step {start_step}")

    # --- 2. Reading the NEW file ---
    # Note: The new file often starts at 0 or at start_step depending on the implementation.
    # Here we assume it contains the continuation. We filter by end_step.
    
    lines_added = 0
    with open(new_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None) # Skip header
        
        for row in reader:
            if not row: continue
            try:                
                # Condition: Must be before the end_step
                if start_step + lines_added < end_step if end_step is not None else True:
                    kept_lines.append(",".join(row) + "\n")
                    lines_added += 1
            except ValueError:
                continue

    print(f"  -> New data ajouté : {lines_added} lignes.")

    # --- 3. Ecriture ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.writelines(kept_lines)
    print(f"  -> Sauvegardé sous : {save_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Manual merge of interrupted training logs.")
    
    # Required arguments for paths
    parser.add_argument("--old-monitor", type=str, required=True, help="Path to the old monitor.csv")
    parser.add_argument("--new-monitor", type=str, required=True, help="Path to the new monitor.csv (continuation)")
    parser.add_argument("--save-monitor", type=str, required=True, help="Output path for the merged monitor")
    
    parser.add_argument("--old-noise", type=str, required=True, help="Path to the old noise.csv")
    parser.add_argument("--new-noise", type=str, required=True, help="Path to the new noise.csv (continuation)")
    parser.add_argument("--save-noise", type=str, required=True, help="Output path for the merged noise")
    
    # Step arguments
    parser.add_argument("--start-step", type=int, required=True, 
                        help="Checkpoint step (cutoff point for the old file).")
    parser.add_argument("--end-step", type=int, default=None, 
                        help="Desired final step (cutoff point for the new file). Optional.")

    args = parser.parse_args()

    print("=== Starting manual merge ===")
    print(f"Start Step (Cutoff Old): {args.start_step}")
    print(f"End Step   (Cutoff New): {args.end_step if args.end_step else 'Unlimited'}")
    print("-" * 40)

    # Merge Monitor
    merge_monitor_logs(
        args.old_monitor, 
        args.new_monitor, 
        args.save_monitor, 
        args.start_step, 
        args.end_step
    )

    # Merge Noise
    merge_noise_logs(
        args.old_noise, 
        args.new_noise, 
        args.save_noise, 
        args.start_step, 
        args.end_step
    )
    
    print("=== Merge completed ===")

if __name__ == "__main__":
    main()