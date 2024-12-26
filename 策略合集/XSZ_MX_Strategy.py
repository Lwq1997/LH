
#-*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码

# 用到策略及数据相关API请加入下面的语句(如果要兼容研究使用可以使用 try except导入
from kuanke.user_space_api import *
from 策略合集.Strategy import Strategy
from jqdata import *
from jqfactor import get_factor_values
from kuanke.wizard import *
import pandas as pd
from jqlib.technical_analysis import *


# 小市值策略
class XSZ_MX_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.new_days = 375  # 已上市天数
        self.factor_list = [
            (  ###
                [
                    'non_recurring_gain_loss',
                    'non_operating_net_profit_ttm',
                    'roe_ttm_8y',
                    'sharpe_ratio_20'
                ],
                [
                    -1.3651516084272432e-13,
                    -3.673549665003535e-14,
                    -0.006872269236387061,
                    -3.922028093095638e-12
                ]
            ),
        ]

    def select(self, context):
        # 空仓期控制
        if self.use_empty_month and context.current_dt.month in (self.empty_month):
            return
        # 止损期控制
        if self.stoplost_date is not None:
            return
        self.select_list = self.__get_rank(context)[:self.max_select_count]
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        initial_list = self.stockpool(context)
        initial_list = self.filter_new_stock(context, initial_list, self.new_days)
        # initial_list = self.filter_locked_shares(context, initial_list, 120)    # 过滤即将大幅解禁

        final_list = []
        # MS
        for factor_list, coef_list in self.factor_list:
            factor_values = get_factor_values(initial_list, factor_list, end_date=context.previous_date, count=1)
            df = pd.DataFrame(index=initial_list, columns=factor_values.keys())
            for i in range(len(factor_list)):
                df[factor_list[i]] = list(factor_values[factor_list[i]].T.iloc[:, 0])
            df = df.dropna()

            df['total_score'] = 0
            for i in range(len(factor_list)):
                df['total_score'] += coef_list[i] * df[factor_list[i]]
            df = df.sort_values(by=['total_score'], ascending=False)  # 分数越高即预测未来收益越高，排序默认降序
            complex_factor_list = list(df.index)[:int(0.1 * len(list(df.index)))]
            q = query(
                valuation.code, valuation.circulating_market_cap, indicator.eps
            ).filter(
                valuation.code.in_(complex_factor_list)
            )
            # .order_by(
            #     valuation.circulating_market_cap.asc()
            # )
            df = get_fundamentals(q)
            df = df[df['eps'] > 0]
            lst = list(df.code)
            final_list = list(set(final_list + lst))

        # 再做一次市值过滤
        q = query(valuation.code). \
            filter(valuation.code.in_(final_list)). \
            order_by(valuation.circulating_market_cap.asc())
        df = get_fundamentals(q)
        final_list = list(df.code)
        final_list = final_list[:min(self.max_select_count, len(final_list))]
        return final_list
