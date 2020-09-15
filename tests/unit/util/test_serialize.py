import pytest_check as check
import impl.util.serialize as serialize_util
from impl.caligraph.util import NAMESPACE_CLG_RESOURCE as CLGR


def test_resource_encoding():
    res = f'{CLGR}12"/50_caliber_Mark_8_gun'
    res_encoded = f'<{CLGR}12%22%2F50_caliber_Mark_8_gun>'
    check.equal(serialize_util._resource_to_string(res), res_encoded)


def test_label_encoding():
    label = '12"/50 caliber Mark 8 gun'
    label_encoded = r'12\"/50 caliber Mark 8 gun'
    check.equal(serialize_util._encode_literal_string(label), label_encoded)
