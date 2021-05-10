# MODEL ENVIRONMENT
MAIN_VER = 120
# 91 - 예전에 만든 모델 다시 돌려본 것
# 92 - data 다시 전처리하고 다시 돌려본 것 (input=4) - better than input=3 (leaky_relu)
# 94 - data 다시 전처리하고 다시 돌려본 것 (input=4) - (relu)
# 96 - leaky_relu, input=3, pred_dim = 4
# 97 - leaky_relu, input=3, pred_dim = 2
# 98 - pred_dim = 4, no dropout, layer normalize (good change)
# 102 - pred_dim = 4, lstm stacked 3 (okay)
# 104 - 102 with more epoch (2nd best)
# 105 - relative (bad)
# 106 - not relative, weighted delta
# TODO : 107 - 1:6 (2nd best)
# 108 - 1:12
# 109 - 1:19
# 110 - 3:1
# 111 - delta만 예측 (bad)
# 112 - -0.5~0.5 => -1~1하도록 x2함 (DELTA X - OKAY, DELTA - 이상)
# 113 - -0.5~0.5, MAIN_BODY, SATELLITE 구분해서 NORMALIZE (BAD)
# 114 - 107재현
# 115 - (3, 2) => (4,)
# 116 - angle
# 117 - loc_rel
# 119 - 107 ver but bidirectional (TODO : best)
# 120 - 119 BUT MSE

SAT_VER = 36
# 8 - data 다시 전처리하고 direct, relu, input=4 로 다시 돌려본 것
# 9 - relative, leaky_relu, input=3로 전처리 후 첫 시도
# 11 - collision data 비율 높이고, stacked 3 lstm + 1 dense로 시도
# 12 - extra big 모두 포함한 train data (33000)
# 13 - 부서지는 원인 찾기 위해 일반 DATA로 다시 시도 (괜찮아짐)
# 14 - TRAIN DATA 더 추가해서 5만개로 돌리되 COLLISION WEIGHTED (다시 이상해짐)
# 15 - FULL DATA로 6만개 input frame=3 (TODO: 4TH BEST)
# 16 - input frame =6로 변경
# 17 - input_frame=3, chamfer + abs loss function 사용
# 18 - GNN (Convolution)
# 19 - GNN (Convolution) parameter up
# 20  -GNN (convolution) removed 1 maxpool - 그나마 15이랑 loss는 비슷 but 결과 뽑으면 추세는 따라가지만 부서짐
# 21 - NOT GNN, but location rel, not chamfer
# 22 - NOT GNN, loc_delta (similar with 21 if updated by coord) delta로 update시 bad
# 23 - 20번 버전에서 loss에 chamfer 추가 (20보다 추세는 좀 더 잘따라가지만 굵어짐)
# 24 - NOT GNN, but location rel, with chamfer 0.05 (21에 chamfer만 추가) (TODO : 3RD BEST)
# 25 - GNN, chamfer0.05, location_rel로 돌림 (23번은 angle_relative엿음)
# 26 - independent GNN
# 27 - independent GNN (epoch만 늘려서)
# 28 - independent GNN LSTM 1개 추가
# 30 - TODO : 2ND BEST
# 32 - SATELLITE18 (2ND BEST)
# 33 - Bidirectional (TODO : 1.5 BEST)
# 34 - 반대 sat 추가 (12)
# 35 - mse (TODO:BEST)

CHAMFER_VER = 48
# encoder best - XX (tf version 2.3 기준으로는 : 33)
# decoder best - 36 (tf version 2.3 기준으로는 : 17)
# prediction best using Point Net - 37 (tf version 2.1 기준)
# prediction best using ConvLSTM2D - 22 (tf version 2.3 기준)
# 48부터 satellite prediction model

NUM_POINTS = 21

MAX_VAL = 18.7062  # Satellite coordinate
MIN_VAL = 1.2938   # Satellite coordinate
MAX_DELTA_MAG = 0.010992166502033163
MAX_MAG_REL_DIST = 1.3091  # maximum relative distance magnitude of all satellites (-1.3091)
MAX_MAG_LOC_REL = 2.5177000000000005 # (2.4886999999999997, -2.5177000000000005)

MAIN_MAX_VAL = 17.7027
MAIN_MIN_VAL = 2.323
MAX_DIST = 0.011002137275362468
MIN_DIST = 2.297213480048632e-05

# MAXIMUMS = [17.6567, 17.6799, 18.6915, 18.706, 18.7005, 18.7061, 18.7061, 18.706, 18.7054, 18.706, 18.7062, 18.7062, 18.706, 18.7004, 18.7062, 18.396, 18.7061, 17.8169, 18.7062, 17.2884, 18.7061, 16.6813, 18.6912, 16.1233, 18.1819, 16.6888, 17.6285, 17.2942, 17.0294, 17.8247, 16.4325, 18.3999, 16.5093, 18.704, 17.0967, 18.7023, 17.6859, 18.7061, 18.2119, 18.7062, 18.6995, 18.7062]
# MINIMUMS = [2.3592, 2.323, 1.3134, 3.9839, 1.5896, 3.8069, 2.1805, 3.3281, 2.7324, 2.7264, 3.3571, 2.1946, 3.929, 1.6229, 3.5806, 1.2958, 3.0348, 1.2942, 2.4237, 1.2938, 1.8437, 1.2939, 1.3349, 1.294, 1.2946, 1.294, 1.2938, 1.2938, 1.2939, 1.2946, 1.2944, 1.2958, 1.2938, 1.5795, 1.2938, 2.084, 1.2938, 2.5445, 1.2945, 3.1504, 1.2951, 3.6947]

ENCODER_EPOCH = 120
ENCODER_BS = 16
ENCODER_LR = 0.001
ENCODER_DR = 0.5
ENCODER_PATIENCE = 90

DECODER_EPOCH = 150
DECODER_BS = 16
DECODER_LR = 0.001
DECODER_DR = 0.2
DECODER_PATIENCE = 90

PRED_MODEL_EPOCH = 350
PRED_MODEL_BS = 128
PRED_MODEL_LR = 0.001
PRED_MODEL_DR = 0.1
PRED_MODEL_PATIENCE = 60

SAT_PRED_MODEL_EPOCH = 450
SAT_PRED_MODEL_BS = 128
SAT_PRED_MODEL_LR = 0.001
SAT_PRED_MODEL_DR = 0.2
SAT_PRED_MODEL_PATIENCE = 70

MAIN_SAVE_PATH = '../../../result/MainBody/version_{}'.format(MAIN_VER)
SAT_SAVE_PATH = '../../../result/Satellite/version_{}'.format(SAT_VER)
CHAMFER_SAVE_PATH = '../../../result/Chamfer/version_{}'.format(CHAMFER_VER)

ANGLE_LST = [15, 60, 90, 150, 225, 285, 310, 340, 30, 110, 140, 195, 250, 270, 300, 325]
ITERATION_LST = [0, 55, 167, 297, 312, 489, 10, 35, 150, 200, 350, 450, 550, 605, 20, 100, 250, 400, 500, 600]

RANDOM_SEED = 2

#============== IMG_DATA_PROCESSED_SCOPE ===============

EXTRA_FOR_BIG = [(310, 200), (310, 450), (30, 605), (15, 35), (285, 605), (270, 450), (340, 450), (250, 550), (60, 150),
                 (325, 550), (15, 450), (60, 550), (285, 150), (30, 200), (250, 605), (310, 10), (340, 150), (325, 35),
                 (270, 200), (140, 605), (90, 150), (140, 200), (250, 35), (285, 450), (195, 200), (285, 200), (60, 35),
                 (140, 35), (195, 550), (150, 200), (195, 605), (225, 35), (310, 605), (285, 35), (15, 150), (270, 605),
                 (225, 550), (15, 200), (30, 450), (110, 35), (140, 450), (30, 150), (270, 150), (325, 200), (60, 10),
                 (225, 150), (225, 200), (195, 10), (90, 550), (110, 150), (90, 605), (250, 150), (195, 35), (15, 10),
                 (150, 35), (150, 450), (150, 550), (60, 450), (250, 10), (60, 605), (140, 150), (300, 35), (110, 605),
                 (30, 35), (150, 605), (15, 605), (285, 550), (30, 10), (325, 10), (225, 450), (325, 450), (195, 450),
                 (225, 605), (300, 10), (340, 35), (110, 450), (150, 10), (150, 150), (340, 550), (340, 605), (310, 35),
                 (195, 150), (30, 550), (325, 150), (310, 150), (225, 10), (300, 200), (270, 35), (300, 605), (250, 200),
                 (15, 550), (110, 200), (90, 450), (285, 10), (140, 10), (310, 550), (90, 10), (300, 150), (90, 200), (325, 605),
                 (340, 200), (110, 10), (270, 550), (110, 550), (140, 550), (250, 450), (300, 450), (270, 10), (90, 35), (60, 200),
                 (300, 550), (340, 10)]

EXTRA_FOR_SAT = [(310, 350), (90, 350), (285, 20), (270, 55), (250, 55), (110, 297), (60, 100), (150, 350), (325, 489), (90, 250),
                 (300, 350), (140, 312), (310, 20), (225, 400), (90, 20), (140, 489), (300, 489), (225, 100), (110, 167), (195, 167),
                 (340, 250), (30, 312), (150, 400), (140, 55), (340, 600), (325, 500), (300, 0), (270, 350), (60, 20), (310, 600), (150, 20),
                 (340, 350), (140, 0), (110, 350), (225, 500), (15, 100), (30, 55), (270, 312), (15, 250), (90, 600), (110, 312), (250, 489),
                 (340, 500), (325, 312), (325, 350), (325, 297), (15, 20), (270, 489), (285, 500), (150, 500), (15, 350), (225, 20), (285, 400),
                 (340, 20), (195, 0), (60, 400), (285, 350), (30, 0), (15, 600), (30, 600), (310, 500), (15, 400), (270, 297), (250, 297),
                 (300, 167), (250, 312), (30, 489), (250, 0), (250, 167), (310, 250), (225, 350), (310, 400), (90, 500), (140, 350), (150, 250)]

COLLISION_INFO = {(15, 0) : (78, 125),
                  (15, 167) : (58, 140),
                  (15, 489) : (15, 80),
                  (30, 20) : (30, 135),
                  (30, 100) : (70, 128),
                  (30, 500) : (15, 75),
                  (60, 55) : (69, 195),
                  (60, 167) : (25, 170),
                  (60, 297) : (10, 130),
                  (60, 312) : (45, 130),
                  (60, 489) : (35, 120),
                  (90, 0) : (73, 137),
                  (90, 55) : (59, 120),
                  (90, 167) : (23, 80),
                  (90, 297) : (9, 65),
                  (90, 312) : (36, 90),
                  (90, 489) : (30, 89),
                  (110, 20) : (0, 100),
                  (110, 100) : (46, 150),
                  (110, 250) : (75, 180),
                  (110, 500) : (78, 150),
                  (110, 600) : (79, 140),
                  (140, 100) : (20, 175),
                  (140, 250) : (40, 190),
                  (140, 400) : (60, 185),
                  (140, 500) : (80, 180),
                  (140, 600) : (95, 195),
                  (150, 0) : (0, 75),
                  (150, 55) : (10, 65),
                  (150, 297) : (15, 115),
                  (150, 312) : (42, 150),
                  (195, 20) : (0, 80),
                  (195, 100) : (0, 90),
                  (195, 400) : (15, 130),
                  (195, 500) : (15, 145),
                  (195, 600) : (15, 150),
                  (225, 0) : (0, 70),
                  (225, 55) : (10, 100),
                  (225, 167) : (25, 140),
                  (225, 297) : (49, 156),
                  (225, 312) : (53, 140),
                  (225, 489) : (60, 170),
                  (250, 20) : (0, 140),
                  (250, 100) : (0, 130),
                  (250, 250) : (0, 100),
                  (250, 400) : (0, 100),
                  (250, 600) : (0, 80),
                  (270, 20) : (60, 150),
                  (270, 400) : (0, 90),
                  (270, 600) : (0, 100),
                  (285, 0) : (0, 90),
                  (285, 55) : (15, 100),
                  (285, 297) : (65, 150),
                  (285, 312) : (37, 130),
                  (285, 489) : (45, 150),
                  (300, 20) : (70, 210),
                  (300, 250) : (0, 150),
                  (300, 500) : (0, 100),
                  (300, 600) : (0, 110),
                  (310, 0) : (0, 70),
                  (310, 55) : (20, 100),
                  (310, 297) : (65, 170),
                  (310, 312) : (52, 150),
                  (310, 489) : (27, 150),
                  (325, 20) : (90, 195),
                  (325, 100) : (0, 130),
                  (325, 250) : (0, 110),
                  (325, 400) : (0, 100),
                  (340, 55) : (50, 150),
                  (340, 167) : (57, 140),
                  (340, 297) : (45, 125),
                  (340, 312) : (40, 170),
                  (340, 489) : (17, 75)}

TRAIN_ANGLE_ITER_PAIRS = [(15, 0), (60, 55), (150, 297), (150, 0), (15, 297), (60, 312), (285, 312), (340, 55),
                          (225, 167), (225, 55), (15, 489), (90, 312), (285, 55), (310, 489), (225, 312), (340, 489),
                          (90, 489), (90, 55), (60, 167), (15, 167), (60, 489), (90, 0), (150, 312), (225, 489),
                          (285, 297), (310, 0), (340, 312), (310, 312), (90, 167), (60, 297), (90, 297), (340, 167),
                          (310, 55), (225, 297), (310, 297), (150, 55), (225, 0), (340, 297), (285, 489), (285, 0),
                          (30, 100), (300, 500), (110, 100), (140, 400),
                          (325, 250), (140, 500), (195, 100), (250, 250), (325, 400), (250, 600), (195, 20), (270, 400),
                          (30, 20), (110, 250), (325, 100), (30, 500), (110, 600), (300, 20), (140, 250),
                          (250, 400), (195, 400), (270, 600), (110, 20), (110, 500), (140, 100), (140, 600),
                          (195, 500), (195, 600), (250, 20), (250, 100), (270, 20), (300, 250), (300, 600), (325, 20)]

for pair in TRAIN_ANGLE_ITER_PAIRS:
    assert pair not in EXTRA_FOR_BIG

VAL_ANGLE_ITER_PAIRS = [(15, 55), (150, 489), (340, 0), (195, 250), (250, 500), (30, 250)]
for pair in VAL_ANGLE_ITER_PAIRS:
    assert pair not in TRAIN_ANGLE_ITER_PAIRS
    assert pair not in EXTRA_FOR_BIG

PURE_TEST_ANGLE_ITER_PAIRS = [(310, 167), (15, 312), (60, 0), (285, 167), (30, 400), (300, 100), (270, 500), (140, 20)]
for pair in PURE_TEST_ANGLE_ITER_PAIRS:
    assert pair not in TRAIN_ANGLE_ITER_PAIRS
    assert pair not in EXTRA_FOR_BIG

TEST_ANGLE_ITER_PAIRS = TRAIN_ANGLE_ITER_PAIRS[:4] + VAL_ANGLE_ITER_PAIRS + PURE_TEST_ANGLE_ITER_PAIRS
