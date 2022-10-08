GRAMMER_CLASS = {
    "CELL(":0,
    "CELL_VALUE(":1,
    "SPAN(":2,
    "VALUE(":3,
    "COUNT(":4,
    "ARGMAX(":5,
    "ARGMIN(":6,
    "KEY_VALUE(":7,
#     "ROW_KEY_VALUE(":7,
#     "COL_KEY_VALUE(":8,
    "SUM(":8,
    "DIFF(":9,
    "DIV(":10,
    "AVG(":11,
    "CHANGE_R(":12,
    "MULTI-SPAN(":13,
    "TIMES(":14,
    ")":15,
#     "[":17,
#     "]":18,
} 

AUX_NUM = {
    "0":16,
    "1":17
}

SCALE_CLASS={
    
#     SCALE = ["", "thousand", "million", "billion", "percent"]
    "THOUNSAND(":18,
    "MILLION(":19,
    "BILLION(":20,
    "PERCENT(":21,
    "NONE(":22,
}



GRAMMER_ID = dict(zip(GRAMMER_CLASS.values(), GRAMMER_CLASS.keys()))
AUX_NUM_ID =  dict(zip(AUX_NUM.values(), AUX_NUM.keys()))
OP_ID = {**GRAMMER_ID, **AUX_NUM_ID}


SCALE = ["", "thousand", "million", "billion", "percent"]
SCALECLASS = ["NONE(", "THOUNSAND(", "MILLION(", "BILLION(", "PERCENT("]
SCALE2CLASS = dict(zip(SCALE, SCALECLASS))
# GRAMMER_ID = dict(zip(GRAMMER_CLASS.values(), GRAMMER_CLASS.keys()))