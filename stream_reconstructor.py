import polars as pl
import numpy as np
import math
from typing import List, Dict, Tuple, Any, Optional

class StreamReconstructor:
    def __init__(
        self, 
        tokenizer_vocab: Dict[str, int],
        scaling_params: Optional[Dict[str, Any]] = None,
        bins: Optional[Dict[str, Any]] = None,
    ):
        """
        args:
            tokenizer_vocab: Dict mapping token strings to IDs.
            scaling_params: (Continuous Mode) Dict of params keyed by token STRING.
            bins: (Discrete Mode) Dict of bin edges keyed by token STRING.
        """
        self.vocab = tokenizer_vocab
        self.id_to_token = {v: k for k, v in tokenizer_vocab.items()}
        
        # Initialize attributes to prevent AttributeErrors
        self.scaling_params_raw = {}
        self.scaling_params_by_id = {}
        self.time_token_id = self.vocab.get('<TIME>', -1)
        self.num_token_id = self.vocab.get('<NUM>', -1)
        
        # Determine Mode
        if scaling_params is not None:
            self.mode = "continuous"
            self._init_continuous(scaling_params)
        else:
            self.mode = "discrete"
            if bins is None:
                raise ValueError("If scaling_params is None (Discrete Mode), 'bins' must be provided.")
            self._init_discrete(bins)

    def reconstruct(
        self, 
        tokens: List[int], 
        values: Optional[List[float]] = None, 
        start_time_us: int = 0
    ) -> Tuple[List[int], List[int], List[float]]:
        """
        Unified entry point. 
        Returns: (timestamps_us, token_ids, numeric_values)
        """
        if self.mode == "continuous":
            if values is None:
                raise ValueError("Continuous mode requires a 'values' list.")
            return self._reconstruct_continuous(tokens, values, start_time_us)
        else:
            return self._reconstruct_discrete(tokens, start_time_us)

    # ==========================================
    #            CONTINUOUS LOGIC
    # ==========================================
    def _init_continuous(self, scaling_params):
        # 1. Store Raw Params (Required for "Stitch and Lookup" of BPE strings)
        self.scaling_params_raw = scaling_params
        
        # 2. Store ID-based Params (Optional optimization for atomic tokens)
        self.scaling_params_by_id = {}
        for code_str, params in scaling_params.items():
            if code_str in self.vocab:
                tid = self.vocab[code_str]
                self.scaling_params_by_id[tid] = params
        
        # Ensure <TIME> is mapped if it exists
        if "<TIME>" in scaling_params and self.time_token_id != -1:
            self.scaling_params_by_id[self.time_token_id] = scaling_params["<TIME>"]

    def _unscale_val_manual(self, params, val):
        """Helper to unscale when we have the parameter dict directly"""
        if not params or val is None or np.isnan(val):
            return val
        
        dist = params.get("distribution", "normal")
        eps = 1e-8

        if dist == "normal":
            return val * math.sqrt(max(params["var"], eps)) + params["mean"]
        elif dist == "lognormal":
            return math.exp(val * math.sqrt(max(params["sigma2"], eps)) + params["mu"])
        elif dist == "gamma":
            mean = params["alpha"] * params["beta"]
            std = math.sqrt(max(params["alpha"] * (params["beta"]**2), eps))
            return val * std + mean
        return val

    def _reconstruct_continuous(self, tokens, values, start_time_us):
        # Result lists
        rec_times = []
        rec_ids = []
        rec_vals = []
        
        current_time_us = start_time_us
        
        # Buffers to hold the current BPE sequence
        token_id_buffer = [] 
        token_str_buffer = [] 
        
        # Pre-fetch for speed
        vocab_get = self.vocab.get
        id2str = self.id_to_token
        time_tid = self.time_token_id
        num_tid = self.num_token_id
        scaling_params = self.scaling_params_raw # Use the raw string-keyed dict
        
        for i, (tid, val) in enumerate(zip(tokens, values)):
            
            # --- 1. TIME TOKEN ---
            if tid == time_tid:
                # Flush any pending buffer (e.g. a code with no value)
                if token_id_buffer:
                    rec_times.extend([current_time_us] * len(token_id_buffer))
                    rec_ids.extend(token_id_buffer)
                    rec_vals.extend([None] * len(token_id_buffer))
                    token_id_buffer = []
                    token_str_buffer = []

                # Calculate Delta
                # Lookahead for Factored Time
                delta = 0.0
                if i + 1 < len(tokens) and tokens[i+1] == num_tid:
                    raw_val = values[i+1]
                    params = scaling_params.get("<TIME>")
                    delta = self._unscale_val_manual(params, raw_val)
                elif val is not None and not np.isnan(val):
                    # Fused Time
                    params = scaling_params.get("<TIME>")
                    delta = self._unscale_val_manual(params, val)
                
                current_time_us += int(delta * 1_000_000)
                continue

            # --- 2. NUM TOKEN ---
            elif tid == num_tid:
                if not token_id_buffer:
                    continue # Orphaned <NUM>
                
                # RECONSTRUCT THE KEY
                # Stitch string parts to match the key in scaling_params
                full_code_str = "".join(token_str_buffer)
                
                # Lookup params using the string key
                params = scaling_params.get(full_code_str)
                
                # Unscale
                final_val = self._unscale_val_manual(params, val)
                
                # Commit the sequence
                rec_times.extend([current_time_us] * len(token_id_buffer))
                rec_ids.extend(token_id_buffer)
                
                # Attach value to the last token of the sequence
                vals_to_add = [None] * (len(token_id_buffer) - 1) + [final_val]
                rec_vals.extend(vals_to_add)
                
                # Clear buffers
                token_id_buffer = []
                token_str_buffer = []

            # --- 3. STANDARD TOKEN ---
            else:
                token_id_buffer.append(tid)
                token_str_buffer.append(id2str.get(tid, ""))

        # Flush leftovers
        if token_id_buffer:
            rec_times.extend([current_time_us] * len(token_id_buffer))
            rec_ids.extend(token_id_buffer)
            rec_vals.extend([None] * len(token_id_buffer))

        return rec_times, rec_ids, rec_vals

    # ==========================================
    #             DISCRETE LOGIC
    # ==========================================
    def _init_discrete(self, bins):
        self.discrete_maps = {}  
        self.time_maps = {}      
        self.q_tokens = {}       

        # 1. Q-Token Detection (Standard)
        for s, tid in self.vocab.items():
            # Handle potential tokenizer prefixes like 'ĠQ1' or ' Q1'
            clean_s = s.replace('Ġ', '').replace(' ', '').strip('[]')
            if clean_s.startswith("Q") and clean_s[1:].isdigit():
                self.q_tokens[int(clean_s[1:])] = tid

        if not self.q_tokens:
             raise ValueError("No Q-tokens (Q1, Q2...) found in vocab.")

        # Helper to calculate midpoints
        def get_mids(edges):
            bounds = [0.0] + list(edges) + [edges[-1]] 
            return [(bounds[i] + bounds[i+1])/2 for i in range(len(bounds)-1)]

        # 2. Define Known Time Flag Variations
        # These are the tokens we look for in the VOCAB
        known_time_flags = {
            "TIME_UNDER_24", "TIME_OVER_24", 
            "<TIME_UNDER_24>", "<TIME_OVER_24>",
            "<TIME>", "TIME"
        }

        # 3. Process Bins
        for bin_key, info in bins.items():
            mids = get_mids(info['edges'])
            
            # Generate the Q-map for this bin definition
            q_map = {}
            for q_idx, q_tid in self.q_tokens.items():
                if q_idx <= len(mids):
                    q_map[q_tid] = mids[q_idx-1]

            # --- LOGIC UPDATE START ---
            # If the bin key is generic "<TIME>", we apply it to ALL found time flags
            if bin_key == "<TIME>":
                mapped_any = False
                for vocab_str in known_time_flags:
                    if vocab_str in self.vocab:
                        flag_id = self.vocab[vocab_str]
                        self.time_maps[flag_id] = q_map
                        mapped_any = True
                        # print(f"DEBUG: Mapped generic <TIME> config to vocab token '{vocab_str}'")
                
                if not mapped_any:
                    print("WARNING: Config has <TIME>, but no time tokens (TIME_UNDER_24, etc.) found in vocab.")
            
            # Otherwise, treat it as a specific key (e.g., "HR", "BP", "TIME_UNDER_24")
            else:
                if bin_key in self.vocab:
                    flag_id = self.vocab[bin_key]
                    
                    # Check if this specific key is actually a time flag
                    if bin_key in known_time_flags:
                        self.time_maps[flag_id] = q_map
                    else:
                        self.discrete_maps[flag_id] = q_map
                else:
                    # Optional: handle mismatch warnings here
                    pass
            # --- LOGIC UPDATE END ---

        self.time_flag_ids = list(self.time_maps.keys())
        self.numeric_flag_ids = list(self.discrete_maps.keys())

    def _reconstruct_discrete(self, tokens, start_time_us):
        # 1. Flatten the time_maps into a Lookup DataFrame
        # This creates a table of [Previous_Token (Flag) | Current_Token (Q) | Value]
        lookup_data = []
        for flag_id, q_map in self.time_maps.items():
            for q_tid, seconds in q_map.items():
                lookup_data.append({
                    "prev_token": flag_id, 
                    "token": q_tid, 
                    "seconds_delta": float(seconds)
                })
        
        # Debug: warning if no maps were created (likely a Vocab key mismatch)
        if not lookup_data:
            print(f"Warning: No time mappings created. Check that 'bins' keys match vocab strings.")
            lookup_df = pl.DataFrame(schema={
                "prev_token": pl.Int64, "token": pl.Int64, "seconds_delta": pl.Float64
            })
        else:
            # Explicit schema ensures types are correct for the join
            lookup_df = pl.DataFrame(lookup_data, schema={
                "prev_token": pl.Int64, "token": pl.Int64, "seconds_delta": pl.Float64
            })

        # 2. Prepare the Main DataFrame
        df = pl.DataFrame({"token": tokens})
        # Create the 'context' column (the token before the current one)
        df = df.with_columns(pl.col("token").shift(1).alias("prev_token"))

        # 3. Perform the Join
        # matches rows where (prev_token == TimeFlag) AND (token == Q_Token)
        df = df.join(
            lookup_df, 
            on=["prev_token", "token"], 
            how="left"
        )

        # 4. Fill Nulls (Non-time tokens get 0.0 delta)
        df = df.with_columns(pl.col("seconds_delta").fill_null(0.0))

        # 5. Calculate Absolute Time
        df = df.with_columns(
            (pl.lit(start_time_us) + (pl.col("seconds_delta").cum_sum() * 1_000_000).cast(pl.Int64)).alias("abs_time_us")
        )

        # 6. Value Logic (Optimized using the same Join strategy)
        # We can optimize the numeric value lookup exactly the same way
        val_lookup_data = []
        for flag_id, q_map in self.discrete_maps.items():
            for q_tid, val in q_map.items():
                val_lookup_data.append({
                    "token": flag_id,      # For values, Flag is Current
                    "next_token": q_tid,   # Q is Next
                    "numeric_value": float(val)
                })
        
        if not val_lookup_data:
             val_lookup_df = pl.DataFrame(schema={
                "token": pl.Int64, "next_token": pl.Int64, "numeric_value": pl.Float64
            })
        else:
            val_lookup_df = pl.DataFrame(val_lookup_data, schema={
                "token": pl.Int64, "next_token": pl.Int64, "numeric_value": pl.Float64
            })

        df = df.with_columns(pl.col("token").shift(-1).alias("next_token"))
        
        # Join for values
        df = df.join(
            val_lookup_df,
            on=["token", "next_token"],
            how="left"
        )

        # 7. Filter and Return
        # We identify rows to keep: they must NOT be flags or Q-values used in reconstruction
        # (This logic remains similar but relies on checking set membership)
        df = df.with_columns([
            pl.col("prev_token").is_in(self.time_flag_ids).alias("is_time_val"),
            pl.col("token").is_in(self.time_flag_ids).alias("is_time_flag"),
            pl.col("prev_token").is_in(self.numeric_flag_ids).alias("is_numeric_val")
        ])

        df_clean = df.filter(
            ~pl.col("is_time_flag") & 
            ~pl.col("is_time_val") & 
            ~pl.col("is_numeric_val")
        )

        return (
            df_clean["abs_time_us"].to_list(),
            df_clean["token"].to_list(),
            df_clean["numeric_value"].to_list()
        )