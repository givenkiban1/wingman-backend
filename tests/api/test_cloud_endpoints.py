import pytest
from httpx import AsyncClient
import os

test_params = {'subcategory_no': 6,
                'num_eng': 1,
                'total_seats': 2,
                'afm_hrs': 5835,
                'cert_max_gr_wt': 1670,
                'dprt_time': 2215,
                'power_units': 110,
                'flight_hours_mean': 18,
                'certs_held': 1,
                'second_pilot': 0,
                'site_seeing': 0,
                'air_medical': 0,
                'crew_sex': 0,
                'type_last_insp_100H': 0,
                'type_last_insp_AAIP': 0,
                'type_last_insp_ANNL': 1,
                'type_last_insp_COAW': 0,
                'type_last_insp_COND': 0,
                'type_last_insp_UNK': 0,
                '_AAPL': 0,
                '_AOBV': 0,
                '_BUS ': 0,
                '_FLTS': 0,
                '_INST': 1,
                '_OWRK': 0,
                '_Other': 0,
                '_PERS': 0,
                '_POSI': 0,
                '_UNK': 0,
                'eng_mfgr_CONTINENTAL': 0,
                'eng_mfgr_LYCOMING': 1,
                'eng_mfgr_P&W': 0,
                'eng_mfgr_ROTAX': 0,
                'eng_mfgr_infrequent_sklearn': 0,
                'far_part_091': 1,
                'far_part_infrequent_sklearn': 0,
                'acft_make_BEECH': 0,
                'acft_make_CESSNA': 1,
                'acft_make_PIPER': 0,
                'acft_make_infrequent_sklearn': 0,
                'fixed_retractable_RETR': 0,
                'acft_category_AIR': 1,
                'acft_category_infrequent_sklearn': 0,
                'homebuilt': 0,
                'crew_category_DSTU': 1,
                'crew_category_FLTI': 0,
                'crew_category_PLT': 0,
                'eng_type_REC': 1,
                'eng_type_infrequent_sklearn': 0,
                'carb_fuel_injection_CARB': 1,
                'carb_fuel_injection_FINJ': 0,
                'carb_fuel_injection_UNK': 0,
                'dprt_apt_id': 1,
                'dest_apt_id': 1,
                'flt_plan_filed_IFR': 0,
                'flt_plan_filed_NONE': 1,
                'flt_plan_filed_VFR': 0,
                'pc_profession': 0}

SERVICE_URL = os.environ.get('SERVICE_URL')

@pytest.mark.asyncio
async def test_root_is_up():
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_root_returns_greeting():
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/")
    assert response.json() == {"greeting": "Hello"}


@pytest.mark.asyncio
async def test_predict_is_up():
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/predict", params=test_params)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_is_dict():
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/predict", params=test_params)
    assert isinstance(response.json(), dict)
    assert len(response.json()) == 1


@pytest.mark.asyncio
async def test_predict_has_key():
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/predict", params=test_params)
    assert response.json().get('fare_amount', False)

@pytest.mark.asyncio
async def test_cloud_api_predict():
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/predict", params=test_params)
    assert isinstance(response.json().get('fare_amount'), float)
