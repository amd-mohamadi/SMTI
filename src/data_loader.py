import numpy as np
import pandas as pd
import os
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees, gps2dist_azimuth
from obspy.taup.taup_create import build_taup_model

def get_taup_model(velocity_model_path=None):
    """
    Load or build the TauP model.
    """
    if velocity_model_path is None:
        # Default to forge.tvel in the project root or similar if not specified
        # But user gave a specific path in comments, better to use that or generic 'forge'
        # if available. For now, we expect the user/caller to handle pathing or
        # we look in standard locations.
        velocity_model_path = "/home/a-mohamdi/Projects/MTfit/MTfit/SMCMTI/forge.tvel"

    model_name = os.path.splitext(os.path.basename(velocity_model_path))[0]
    
    # Check if .npz already exists
    npz_path = os.path.join(os.path.dirname(velocity_model_path), f"{model_name}.npz")
    
    if os.path.exists(npz_path):
        return TauPyModel(model=npz_path)
    
    # If not, try to build it (might require write permissions)
    try:
        build_taup_model(velocity_model_path, output_folder=os.path.dirname(velocity_model_path))
        return TauPyModel(model=npz_path)
    except Exception as e:
        print(f"Warning: Could not build TauP model from {velocity_model_path}: {e}")
        # Fallback to standard iasp91 or raise
        raise e

def read_data(
    data_path,
    min_p_phase_score=0.6,
    min_s_phase_score=0.6,
    min_ppl_score=0.6,
    min_shpl_score=0.6,
    min_svpl_score=0.6,
    p_polarity_phase="Pz",
    inversion_options=None,
    velocity_model_path="/home/a-mohamdi/Projects/MTfit/MTfit/SMCMTI/forge.tvel"
):
    """
    Read data.dat file and prepare input dictionary for MT inversion.
    
    Parameters
    ----------
    data_path : str
        Path to data.dat file.
    min_p_phase_score : float
        Minimum phase score for P-wave amplitude measurements.
    min_s_phase_score : float
        Minimum phase score for S-wave (SH/SV) amplitude measurements.
    min_ppl_score : float
        Minimum phase_score for P polarity measurements.
    min_shpl_score : float
        Minimum phase_score for SH polarity measurements.
    min_svpl_score : float
        Minimum phase_score for SV polarity measurements.
    p_polarity_phase : str
        Phase name to use for P polarity (e.g. 'P', 'Pz'). Default 'Pz'.
    inversion_options : list[str] or None
        List of keys to include in the output (e.g. ['PPolarity', 'P/SHAmplitudeRatio']).
        If None, all available keys are returned.
    velocity_model_path : str
        Path to the .tvel or .npz velocity model file for TauP.

    Returns
    -------
    dict
        Dictionary containing formatted data for inversion.
    """
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load data
    try:
        df = pd.read_csv(data_path, sep="\t")
    except Exception:
        # Try whitespace if tab fails
        df = pd.read_csv(data_path, delim_whitespace=True)

    # Required columns check
    required_cols = [
        'station_id', 'phase_type', 'phase_polarity', 'phase_score', 
        'phase_amplitude', 'latitude_pick', 'longitude_pick', 'depth_km_pick',
        'latitude_station', 'longitude_station', 'depth_km_station'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in data.dat: {missing}")

    # Initialize TauP
    taup_model = get_taup_model(velocity_model_path)

    # Initialize containers
    data_dict = {}
    
    # -------------------------------------------------------------------------
    # Helper to calculate Station geometry (Azimuth, TakeOff)
    # -------------------------------------------------------------------------
    station_geom_cache = {}

    def get_geom(row):
        sta_id = row['station_id']
        if sta_id in station_geom_cache:
            return station_geom_cache[sta_id]
        
        # Calculate distance
        dist_deg = locations2degrees(
            row['latitude_pick'], row['longitude_pick'],
            row['latitude_station'], row['longitude_station']
        )
        
        # Calculate Azimuth
        _, az, _ = gps2dist_azimuth(
            row['latitude_pick'], row['longitude_pick'],
            row['latitude_station'], row['longitude_station']
        )
        
        # Calculate TakeOff Angle
        # Logic from 3.create_input_mtfit.py
        d_source = abs(row['depth_km_pick'])
        d_receiver = abs(row['depth_km_station'])
        
        if d_source > d_receiver:
            arrivals = taup_model.get_ray_paths(
                source_depth_in_km=d_source,
                receiver_depth_in_km=d_receiver,
                distance_in_degree=dist_deg,
                phase_list=["P", "p"]
            )
            # Takeoff angle at source
            takeoff = arrivals[0].takeoff_angle if arrivals else 90.0
        else:
            # Source shallower than receiver, use incident_angle? 
            # 3.create_input_mtfit.py uses incident_angle in the 'else' block, 
            # assuming reciprocity or upgoing rays? 
            # Wait, standard MT inversion needs takeoff angle at the SOURCE.
            # If source < receiver, it's a downgoing ray usually?
            # Let's stick to the user's legacy logic exactly.
            arrivals = taup_model.get_ray_paths(
                source_depth_in_km=d_receiver, # Swapped?
                receiver_depth_in_km=d_source, # Swapped?
                distance_in_degree=dist_deg,
                phase_list=["P", "p"]
            )
            # Legacy code uses incident_angle here.
            takeoff = arrivals[0].incident_angle if arrivals else 90.0
            
        takeoff = round(takeoff, 2)
        station_geom_cache[sta_id] = (az, takeoff)
        return az, takeoff

    # -------------------------------------------------------------------------
    # Parse Polarity (PPolarity, SHPolarity, SVPolarity)
    # -------------------------------------------------------------------------
    # Strategy: Group by station, look for specific phases
    
    # Pre-filter for efficiency
    # Use Pz for P-polarity if requested, else P
    # Use Sh for SH, Sv for SV
    
    # We need to process station by station
    stations = df['station_id'].unique()
    
    # Lists to accumulate results
    pol_data = {
        'PPolarity': {'names': [], 'az': [], 'to': [], 'val': [], 'err': []},
        'SHPolarity': {'names': [], 'az': [], 'to': [], 'val': [], 'err': []},
        'SVPolarity': {'names': [], 'az': [], 'to': [], 'val': [], 'err': []},
    }
    
    ratio_data = {
        'P/SHAmplitudeRatio': {'names': [], 'az': [], 'to': [], 'val': [], 'err': []},
        'P/SVAmplitudeRatio': {'names': [], 'az': [], 'to': [], 'val': [], 'err': []},
        'SH/SVAmplitudeRatio': {'names': [], 'az': [], 'to': [], 'val': [], 'err': []},
    }

    # Error constraints
    MAX_POL_ERROR = 0.5  # Cap error? user script used different values per phase
    MAX_AMP_ERROR = 0.5 # Typically 0.5 or user constrained
    
    # Default error caps from legacy script
    MAX_P_POL_ERR = 0.5
    MAX_SH_POL_ERR = 0.5
    MAX_SV_POL_ERR = 0.5
    MAX_RATIO_ERR = 0.5

    for station in stations:
        # if station.startswith("CP.C1"):continue
        rows = df[df['station_id'] == station]
        
        # --- Polarity ---
        # P Polarity
        p_rows = rows[rows['phase_type'] == p_polarity_phase]
        if not p_rows.empty:
            row = p_rows.iloc[0] # Take first if multiple
            # Filter: score
            if (row['phase_score'] >= min_p_phase_score) and (abs(row['phase_polarity']) >= min_ppl_score):
                az, to = get_geom(row)
                meas = 1 if row['phase_polarity'] > 0 else -1
                # Error: min(MAX_ERR, abs(0.9 - abs(polarity)))
                err = min(MAX_P_POL_ERR, abs(1.0 - abs(row['phase_polarity'])))
                
                pol_data['PPolarity']['names'].append(station)
                pol_data['PPolarity']['az'].append(az)
                pol_data['PPolarity']['to'].append(to)
                pol_data['PPolarity']['val'].append(meas)
                pol_data['PPolarity']['err'].append(err)

        # SH Polarity
        sh_rows = rows[rows['phase_type'] == 'Sh']
        if not sh_rows.empty:
            row = sh_rows.iloc[0]
            if (row['phase_score'] >= min_s_phase_score) and (abs(row['phase_polarity']) >= min_shpl_score):
                az, to = get_geom(row)
                meas = 1 if row['phase_polarity'] > 0 else -1
                err = min(MAX_SH_POL_ERR, abs(1.0 - abs(row['phase_polarity'])))
                
                pol_data['SHPolarity']['names'].append(station)
                pol_data['SHPolarity']['az'].append(az)
                pol_data['SHPolarity']['to'].append(to)
                pol_data['SHPolarity']['val'].append(meas)
                pol_data['SHPolarity']['err'].append(err)

        # SV Polarity
        sv_rows = rows[rows['phase_type'] == 'Sv']
        if not sv_rows.empty:
            row = sv_rows.iloc[0]
            if (row['phase_score'] >= min_s_phase_score) and (abs(row['phase_polarity']) >= min_svpl_score):
                az, to = get_geom(row)
                meas = 1 if row['phase_polarity'] > 0 else -1
                err = min(MAX_SV_POL_ERR, abs(1.0 - abs(row['phase_polarity'])))
                
                pol_data['SVPolarity']['names'].append(station)
                pol_data['SVPolarity']['az'].append(az)
                pol_data['SVPolarity']['to'].append(to)
                pol_data['SVPolarity']['val'].append(meas)
                pol_data['SVPolarity']['err'].append(err)

        # --- Amplitude Ratios ---
        # We need Pz, Sh, Sv rows for this, filtered by phase_score only
        # User: "Pz-> PAmplitude, Sh->SHAmplitude, Sv->SVAmplitude"
        # User: "min phase_score value for the amplitude ratio"
        
        # Get best candidates for each phase
        pz = rows[(rows['phase_type'] == 'Pz') & (rows['phase_score'] > min_p_phase_score) & (rows['phase_amplitude'] > 1e-12)]
        sh = rows[(rows['phase_type'] == 'Sh') & (rows['phase_score'] > min_s_phase_score) & (rows['phase_amplitude'] > 1e-12)]
        sv = rows[(rows['phase_type'] == 'Sv') & (rows['phase_score'] > min_s_phase_score) & (rows['phase_amplitude'] > 1e-12)]
        
        # Use first valid pick for each
        p_row = pz.iloc[0] if not pz.empty else None
        sh_row = sh.iloc[0] if not sh.empty else None
        sv_row = sv.iloc[0] if not sv.empty else None
        
        # P/SH
        if p_row is not None and sh_row is not None:
            # Check SNR? Legacy script used SNR > threshold. 
            # We'll rely on input filtered data or phase_score for now as user simplified request.
            # Geometry from P row usually
            az, to = get_geom(p_row) 
            
            p_amp = p_row['phase_amplitude']
            sh_amp = sh_row['phase_amplitude']
            
            # Error Calculation (ABSOLUTE)
            # Legacy: err = min(MAX, 1.01 - score) -> This was RELATIVE/PERCENT error in legacy input?
            # 3.create_input_mtfit.py passed it as 'Error', and Inversion looked at `amplitude_errors_as_percent`
            # For data.dat text format, it was relative.
            # But here we are building the dictionary directly for InversionPyTensor.
            # InversionPyTensor.amplitude_ratio_matrix treats input Error as ABSOLUTE if it's not converted.
            # Wait, `amplitude_ratio_matrix` code:
            # this_perc_err1 = (err_num / np.abs(num_meas))
            # So it EXPECTS `err_num` to be the ABSOLUTE error.
            
            # User logic: Error = min(MAX_AMPLITUDE_ERROR, 1.01-phase_score)
            # This result (e.g. 0.1) is generally treated as 10% or 0.1 fractional error.
            # So to get ABSOLUTE Error, we must multiply by Amplitude.
            
            p_frac_err = min(MAX_RATIO_ERR, 1.1 - p_row['phase_score'])
            sh_frac_err = min(MAX_RATIO_ERR, 1.1 - sh_row['phase_score'])
            
            p_abs_err = p_amp * p_frac_err
            sh_abs_err = sh_amp * sh_frac_err
            
            ratio_data['P/SHAmplitudeRatio']['names'].append(station)
            ratio_data['P/SHAmplitudeRatio']['az'].append(az)
            ratio_data['P/SHAmplitudeRatio']['to'].append(to)
            ratio_data['P/SHAmplitudeRatio']['val'].append([p_amp, sh_amp])
            ratio_data['P/SHAmplitudeRatio']['err'].append([p_abs_err, sh_abs_err])

        # P/SV
        if p_row is not None and sv_row is not None:
            az, to = get_geom(p_row)
            p_amp = p_row['phase_amplitude']
            sv_amp = sv_row['phase_amplitude']
            
            p_frac_err = min(MAX_RATIO_ERR, 1.1 - p_row['phase_score'])
            sv_frac_err = min(MAX_RATIO_ERR, 1.1 - sv_row['phase_score'])
            
            p_abs_err = p_amp * p_frac_err
            sv_abs_err = sv_amp * sv_frac_err
            
            ratio_data['P/SVAmplitudeRatio']['names'].append(station)
            ratio_data['P/SVAmplitudeRatio']['az'].append(az)
            ratio_data['P/SVAmplitudeRatio']['to'].append(to)
            ratio_data['P/SVAmplitudeRatio']['val'].append([p_amp, sv_amp])
            ratio_data['P/SVAmplitudeRatio']['err'].append([p_abs_err, sv_abs_err])

        # SH/SV
        if sh_row is not None and sv_row is not None:
            # Use SH geometry?
            az, to = get_geom(sh_row)
            sh_amp = sh_row['phase_amplitude']
            sv_amp = sv_row['phase_amplitude']
            
            sh_frac_err = min(MAX_RATIO_ERR, 1.1 - sh_row['phase_score'])
            sv_frac_err = min(MAX_RATIO_ERR, 1.1 - sv_row['phase_score'])
            
            sh_abs_err = sh_amp * sh_frac_err
            sv_abs_err = sv_amp * sv_frac_err
            
            ratio_data['SH/SVAmplitudeRatio']['names'].append(station)
            ratio_data['SH/SVAmplitudeRatio']['az'].append(az)
            ratio_data['SH/SVAmplitudeRatio']['to'].append(to)
            ratio_data['SH/SVAmplitudeRatio']['val'].append([sh_amp, sv_amp])
            ratio_data['SH/SVAmplitudeRatio']['err'].append([sh_abs_err, sv_abs_err])

    # -------------------------------------------------------------------------
    # Build Final Dictionary
    # -------------------------------------------------------------------------
    final_data = {}
    
    # Populate UID
    evid = df['event_index'].iloc[0] if 'event_index' in df.columns else 'unknown_event'
    final_data['UID'] = str(evid)

    # Helper to construct component dicts
    def construct_dict(source_dict):
        if not source_dict['names']:
            return None
        
        return {
            'Stations': {
                'Name': source_dict['names'],
                'Azimuth': np.array(source_dict['az']).reshape(-1, 1),
                'TakeOffAngle': np.array(source_dict['to']).reshape(-1, 1),
            },
            'Measured': np.array(source_dict['val']).reshape(-1, 1 if isinstance(source_dict['val'][0], (int, float)) else 2),
            'Error': np.array(source_dict['err']).reshape(-1, 1 if isinstance(source_dict['err'][0], (int, float)) else 2)
        }

    # Polarity
    for key in pol_data:
        if inversion_options is None or key in inversion_options:
            res = construct_dict(pol_data[key])
            if res:
                final_data[key] = res

    # Ratios
    for key in ratio_data:
        if inversion_options is None or key in inversion_options:
            res = construct_dict(ratio_data[key])
            if res:
                final_data[key] = res

    return final_data
