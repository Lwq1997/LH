factor_list = [
    (  # ARBR-SGAI-NPTTOR-RPPS.txt
        [
            'ARBR',  # 情绪类因子 ARBR
            'SGAI',  # 质量类因子 销售管理费用指数
            'net_profit_to_total_operate_revenue_ttm',  # 质量类因子 净利润与营业总收入之比
            'retained_profit_per_share'  # 每股指标因子 每股未分配利润
        ],
        [
            -2.3425,
            -694.7936,
            -170.0463,
            -1362.5762
        ]
    ),
    (  # FL-VOL240-AEttm.txt
        [
            'financial_liability',
            'VOL240',
            'administration_expense_ttm'
        ],
        [
            -5.305338739321596e-13,
            0.0028018907262207246,
            3.445005190225511e-13
        ]
    )]

new_list = [list(factor_list[0])]
print(new_list)

for factor_list, coef_list in new_list:
    print(factor_list, coef_list)

for i in range(13):
    print(i)

import numpy as np
import pandas as pd

# 定义原方法
def __calculate_bollinger_width(prices, period=20, nbdev=2):
    ma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    upper = ma + nbdev * std
    lower = ma - nbdev * std
    if ma == 0:
        return 0
    width = (upper - lower) / ma
    return width

def calculate_RSI( close_prices, period=14):
    # 计算价格变动
    delta = np.diff(close_prices)
    # 分离上涨和下跌的变动
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # 计算平均涨幅和平均跌幅
    avg_gain = []
    avg_loss = []
    for i in range(len(gain)):
        if i < period:
            avg_gain.append(np.sum(gain[0:i + 1]) / (i + 1))
            avg_loss.append(np.sum(loss[0:i + 1]) / (i + 1))
        else:
            avg_gain.append((avg_gain[-1] * (period - 1) + gain[i]) / period)
            avg_loss.append((avg_loss[-1] * (period - 1) + loss[i]) / period)
    # 计算RS和RSI
    rs = np.array(avg_gain) / np.array(avg_loss)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def calculate_RSI_1( prices, period=14):
    """ 计算RSI指标 """
    deltas = np.diff(prices)
    ups = deltas[deltas > 0].sum()
    downs = -deltas[deltas < 0].sum()
    if downs == 0:
        return 100
    rs = ups / downs
    return 100 - (100 / (1 + rs))

# 定义之前提供的方法
def bollinger_band_width(close_prices, window=20, num_std=2):
    ma = pd.Series(close_prices).rolling(window).mean()
    std = pd.Series(close_prices).rolling(window).std()
    upper_band = ma + num_std * std
    lower_band = ma - num_std * std
    band_width = (upper_band - lower_band) / ma
    return band_width

# 示例数据
prices = np.random.rand(100)
print("原始RSI提供方法结果:", calculate_RSI_1(prices))
print("RSI提供方法结果:", calculate_RSI(prices))
print("RSI提供方法结果2:", calculate_RSI(prices)[-1])
print("原方法结果:", __calculate_bollinger_width(prices))
print("提供方法结果:", bollinger_band_width(prices))
print("提供方法结果1:", bollinger_band_width(prices)[len(bollinger_band_width(prices))-1])