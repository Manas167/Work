# ----------------------------------------
# Functions for processing Bloomberg Data
# ----------------------------------------

import numpy as np
import pandas as pd
from ConvertibleBondClass import ConvertibleBond
from AndersonBuffumPricer import AndersenBuffumPricer


# process call put schedule data
def process_call_put_schedule_data(schedule_data):
    ISINs = schedule_data.columns[1::4]
    schedules = {}
    for ISIN in ISINs:
        loc = schedule_data.columns.get_loc(ISIN)
        call = schedule_data.iloc[:,loc:loc+2][1:].dropna()
        call.columns = ['date','price']
        call['date'] = pd.to_datetime(call['date'])
        put = schedule_data.iloc[:,loc+2:loc+4][1:].dropna()
        put.columns = ['date','price']
        put['date'] = pd.to_datetime(put['date'])
        schedules[ISIN] = (call,put)
    return schedules


# parse bloomberg cds spread data
def parse_BBG_cds_spread(ISIN, cds_data):
    try:
        r = cds_data.index[cds_data['ISIN'] == ISIN].tolist()[0] + 1
        data = cds_data.iloc[r]
        sprd = data.values[2:]
        if(any(np.isnan(s) for s in sprd)):
            return np.array([])
        else:
            return sprd
    except:
        return np.array([])


def ABpricer_on_BBG_data(row, pdate, cds_data, cp_schedules, rf_curve, p, use_cds_term_structure=False):
    contract_info = {
        'maturity_date': row['MATURITY'],
        'coupon_rate': row['CPN'],
        'coupon_freq': row['CPN_FREQ'],
        'notional': 1000,
        'conv_ratio': None,  # will be derived from conv_price in the code
        'conv_price': row['CV_CNVS_PX'],
        'callable': row['CALLABLE'] == 'Y',
        'puttable': row['PUTABLE'] == 'Y',
    }

    if (contract_info['callable'] or contract_info['puttable']):
        ISIN = row['ISIN']
        if (ISIN in cp_schedules):
            contract_info['call_schedule'], contract_info['put_schedule'] = cp_schedules[ISIN]
        else:
            contract_info['call_schedule'], contract_info['put_schedule'] = (None, None)

    if 'SOFT CALL 20-30' in row:
        contract_info['softcall'] = row['SOFT CALL 20-30'] == 'Y'
    else:
        contract_info['softcall'] = False

    if (contract_info['softcall']):
        contract_info['softcall_start'] = pd.to_datetime(row['SOFT CALL START'])
        contract_info['softcall_end'] = pd.to_datetime(row['SOFT CALL END'])
        contract_info['softcall_barrier'] = row['SOFT CALL BARRIER']
        contract_info['softcall_redempt'] = row['SOFT CALL REDEMPTION']

    rr = row['BOND_RECOVERY_RATE']
    rr = 0.4 if np.isnan(rr) else rr

    if (use_cds_term_structure == True):
        credit_spread = parse_BBG_cds_spread(row['ISIN'], cds_data)
        credit_tenors = [0.5, 1, 2, 3, 4, 5, 7, 10]
    else:
        credit_spread = [row['FLAT_CREDIT_SPREAD_CV_MODEL']]
        credit_tenors = [5]

    model_params = {
        'recovery_rate': rr,
        'equity_spot': row['CV_MODEL_UNDL_PX'],
        'equity_dividend_yield': row['EQY_DVD_YLD_IND'] / 100.,
        'equity_flat_vol': row['CV_MODEL_STOCK_VOL'],
        'eta': row['STOCK_JUMP_ON_DEFAULT_CV_MODEL'],
        'credit tenors': credit_tenors,
        'credit spread': credit_spread,
    }

    CB = ConvertibleBond(contract_info, model_params, rf_curve, p=p)
    ABpricer = AndersenBuffumPricer(CB, pdate, dt=1. / 48., dy=0.05)

    output = (ABpricer.clean_price(), ABpricer.eq_spot_delta(), ABpricer.eq_spot_gamma(), ABpricer.eq_vega())
    return output


def reformat_output_data(output_data_original, cb_data_input):
    output_data = output_data_original.copy()
    output_data['Issuer'] = cb_data_input['NAME'].values
    output_data['Maturity'] = cb_data_input['MATURITY'].values
    output_data['MKT Price'] = cb_data_input['PX_LAST'].values
    output_data['BBG Price'] = cb_data_input['Fair Value'].values
    output_data['BBG Delta'] = cb_data_input['CV_MODEL_DELTA_V'].values
    output_data['BBG Gamma'] = cb_data_input['CV_MODEL_GAMMA_V'].values
    output_data['BBG Vega'] = cb_data_input['CV_MODEL_VEGA'].values
    output_data['CONV PX'] = cb_data_input['CV_CNVS_PX'].values
    output_data['EQ Spot'] = cb_data_input['CV_MODEL_UNDL_PX'].values
    output_data['EQ Vol'] = cb_data_input['CV_MODEL_STOCK_VOL'].values
    output_data['Flat CDS Spread'] = cb_data_input['FLAT_CREDIT_SPREAD_CV_MODEL'].values

    cols_order = [
        'Issuer', 'Maturity', 'EQ Spot', 'CONV PX', 'EQ Vol', 'Flat CDS Spread',
        'MKT Price', 'BBG Price', 'ABM Price', 'BBG Delta', 'ABM Delta',
        'BBG Gamma', 'ABM Gamma', 'BBG Vega', 'ABM Vega']

    output_data = output_data.loc[:, cols_order]
    output_data['Maturity'] = output_data['Maturity'].astype(str)
    return output_data
